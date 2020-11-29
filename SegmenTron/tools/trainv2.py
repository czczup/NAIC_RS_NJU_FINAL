import time
import copy
import datetime
import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import logging
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from tabulate import tabulate
from torchvision import transforms
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.solver.loss import get_segmentation_loss
from segmentron.solver.optimizer import get_optimizer
from segmentron.solver.lr_scheduler import get_scheduler
from segmentron.utils.distributed import *
from segmentron.utils.score import SegmentationMetric
from segmentron.utils.filesystem import save_checkpoint
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from segmentron.utils.visualize import show_flops_params
from segmentron.config import cfg
try:
    import apex
except:
    pass



class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])
        # dataset and dataloader
        data_kwargs = {'transform': input_transform, 'base_size': cfg.TRAIN.BASE_SIZE,
                       'crop_size': cfg.TRAIN.CROP_SIZE}
        train_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='train', mode='train', **data_kwargs)
        val_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='val', mode="val", **data_kwargs)
        self.iters_per_epoch = len(train_dataset) // (args.num_gpus * cfg.TRAIN.BATCH_SIZE)
        self.max_iters = cfg.TRAIN.EPOCHS * self.iters_per_epoch
        self.classes_A = train_dataset.classes_A
        self.classes_C = train_dataset.classes_C

        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(train_sampler, cfg.TRAIN.BATCH_SIZE, self.max_iters, drop_last=True)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, cfg.TEST.BATCH_SIZE, drop_last=False)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=cfg.DATASET.WORKERS,
                                            pin_memory=True)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=cfg.DATASET.WORKERS,
                                          pin_memory=True)

        # create network
        self.model = get_segmentation_model().to(self.device)
        
        # print params and flops
        if get_rank() == 0:
            try:
                show_flops_params(copy.deepcopy(self.model), args.device)
            except Exception as e:
                logging.warning('get flops and params error: {}'.format(e))


        if cfg.MODEL.BN_TYPE not in ['BN']:
            logging.info('Batch norm type is {}, convert_sync_batchnorm is not effective'.format(cfg.MODEL.BN_TYPE))
        elif args.distributed and cfg.TRAIN.SYNC_BATCH_NORM:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            logging.info('SyncBatchNorm is effective!')
        else:
            logging.info('Not use SyncBatchNorm!')

        # create criterion
        self.criterion = get_segmentation_loss(cfg.MODEL.MODEL_NAME, use_ohem=cfg.SOLVER.OHEM,
                                               aux=cfg.SOLVER.AUX, aux_weight=cfg.SOLVER.AUX_WEIGHT,
                                               ignore_index=cfg.DATASET.IGNORE_INDEX).to(self.device)

        # optimizer, for model just includes encoder, decoder(head and auxlayer).
        self.optimizer = get_optimizer(self.model)

        # lr scheduling
        self.lr_scheduler = get_scheduler(self.optimizer, max_iters=self.max_iters,
                                          iters_per_epoch=self.iters_per_epoch)

        # resume checkpoint if needed
        self.start_epoch = 0

        if args.resume and os.path.isfile(args.resume):
            name, ext = os.path.splitext(args.resume)
            assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
            logging.info('Resuming training, loading {}...'.format(args.resume))
            resume_sate = torch.load(args.resume, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(resume_sate['state_dict'])
            self.start_epoch = resume_sate['epoch']
            logging.info('resume train from epoch: {}'.format(self.start_epoch))
            if resume_sate['optimizer'] is not None and resume_sate['lr_scheduler'] is not None:
                logging.info('resume optimizer and lr scheduler from resume state..')
                self.optimizer.load_state_dict(resume_sate['optimizer'])
                self.lr_scheduler.load_state_dict(resume_sate['lr_scheduler'])

        if cfg.SOLVER.USE_FP16 == True:
            self.model, self.optimizer = apex.amp.initialize(self.model.cuda(), self.optimizer, opt_level="O1")
            logging.info("Initializing mixed precision done.")

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],
                                                             output_device=args.local_rank,
                                                             find_unused_parameters=True)
        else:
            self.model = nn.parallel.DataParallel(self.model)


        # evaluation metrics
        self.metric_A = SegmentationMetric(train_dataset.NUM_CLASS_A, args.distributed)
        self.metric_C = SegmentationMetric(train_dataset.NUM_CLASS_C, args.distributed)

        self.best_pred = 0.0


    def train(self):
        self.save_to_disk = get_rank() == 0
        epochs, max_iters, iters_per_epoch = cfg.TRAIN.EPOCHS, self.max_iters, self.iters_per_epoch
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_epoch * self.iters_per_epoch
        start_time = time.time()
        logging.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))
        self.model.train()

        iteration = self.start_epoch * iters_per_epoch if self.start_epoch > 0 else 0
        for (images, targets_A, targets_C) in self.train_loader:

            epoch = iteration // iters_per_epoch + 1

            iteration += 1
            images = images.to(self.device, non_blocking=True)
            targets_A = targets_A.to(self.device, non_blocking=True)
            targets_C = targets_C.to(self.device, non_blocking=True)

            outputs = self.model(images)
            loss_dict = self.criterion(outputs, tuple([targets_A, targets_C]))

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)

            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            self.optimizer.zero_grad()
            if cfg.SOLVER.USE_FP16 == True:
                with apex.amp.scale_loss(losses, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                losses.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=35, norm_type=2)
            
            try:
                self.optimizer.step()
                self.lr_scheduler.step()
            except:
                pass
            eta_seconds = ((time.time() - start_time) / iteration) * (max_iters - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % log_per_iters == 0 and self.save_to_disk:
                logging.info(
                    "Epoch: {:d}/{:d} || Iters: {:d}/{:d} || Lr: {:.6f} || "
                    "Loss_8: {:.4f} || Loss_14: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                        epoch, epochs, iteration % iters_per_epoch, iters_per_epoch,
                        self.optimizer.param_groups[0]['lr'], loss_dict['loss_8'].item(), loss_dict['loss_14'].item(),
                        str(datetime.timedelta(seconds=int(time.time() - start_time))),
                        eta_string))

            if iteration % self.iters_per_epoch == 0 and self.save_to_disk:
                save_checkpoint(self.model, epoch, self.optimizer, self.lr_scheduler, is_best=False)

            if not self.args.skip_val and iteration % val_per_iters == 0:
                self.validation(epoch)
                self.model.train()

        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logging.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / max_iters))

    def validation(self, epoch):
        self.metric_A.reset()
        self.metric_C.reset()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()
        model.eval()
        for i, (image, target_A, target_C) in enumerate(self.val_loader):
            image = image.to(self.device, non_blocking=True)
            target_A = target_A.to(self.device, non_blocking=True)
            target_C = target_C.to(self.device, non_blocking=True)

            with torch.no_grad():
                outputs = model(image)
                output_A = outputs[0][0]
                output_C = outputs[1][0]

            self.metric_A.update(output_A, target_A)
            self.metric_C.update(output_C, target_C)
            pixAcc_A, mIoU_A = self.metric_A.get()
            pixAcc_C, mIoU_C = self.metric_C.get()

            logging.info("[EVAL] Sample: {:d}, datasetA [pixAcc: {:.3f}, mIoU: {:.3f}], "
                         "datasetC [pixAcc: {:.3f}, mIoU: {:.3f}]"
                         .format(i + 1, pixAcc_A * 100, mIoU_A * 100, pixAcc_C * 100, mIoU_C * 100))
        pixAcc_A, mIoU_A, category_iou_A = self.metric_A.get(return_category_iou=True)
        pixAcc_C, mIoU_C, category_iou_C = self.metric_C.get(return_category_iou=True)

        logging.info("[EVAL END] Epoch: {:d}, datasetA [pixAcc: {:.3f}, mIoU: {:.3f}], "
                     "datasetC [pixAcc: {:.3f}, mIoU: {:.3f}]"
                     .format(epoch, pixAcc_A * 100, mIoU_A * 100, pixAcc_C * 100, mIoU_C * 100))
        
       
        
        weights_A = [0.1075814, 0.06220228, 0.18560777, 0.1683705,
                     0.08614201, 0.15660633, 0.05183542, 0.18680766]
        
        weights_C = [0.02754302, 0.07850080, 0.12062946, 0.06670495,
                     0.00267682, 0.22780647, 0.01259433, 0.12597127,
                     0.02395271, 0.15588569, 0.07321313, 0.04918330,
                     0.01127538, 0.02406267]
        weighted_iou_A = [weights_A[i] * category_iou_A[i] for i in range(len(weights_A))]
        fwiou_A = sum(weighted_iou_A)
        weighted_iou_C = [weights_C[i] * category_iou_C[i] for i in range(len(weights_C))]
        fwiou_C = sum(weighted_iou_C)
        logging.info("DatasetA FWIoU: %.6f, DatasetC FWIoU: %.6f" % (fwiou_A * 100, fwiou_C * 100))
        
        synchronize()
        if self.best_pred < fwiou_C and self.save_to_disk:
            self.best_pred = fwiou_C
            logging.info('Epoch {} is the best model, save the model..'.format(epoch))
            save_checkpoint(model, epoch, is_best=True)


if __name__ == '__main__':
    args = parse_args()
    # get config
    cfg.DIR_NAME = args.config_file.split("/")[-1].split(".")[0]
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'train'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()

    checkpoint_dir = "runs/checkpoints/"+cfg.DIR_NAME
    if os.path.exists(checkpoint_dir):
        checkpoint_names = os.listdir(checkpoint_dir)
        checkpoint_name = None
        for i in range(121):
            if str(i)+".pth" in checkpoint_names:
                checkpoint_name = str(i)+".pth"
        if checkpoint_name != None:
            args.resume = checkpoint_dir + "/" + checkpoint_name
        print("resume:", args.resume)

    default_setup(args)

    # create a trainer and start train
    trainer = Trainer(args)
    trainer.train()
