from __future__ import print_function

import os
import sys
import cv2

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import logging
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import time
from tabulate import tabulate
from torchvision import transforms
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.score import SegmentationMetric
from segmentron.utils.distributed import synchronize, make_data_sampler, make_batch_data_sampler
from segmentron.config import cfg
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup


class Evaluator(object):
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
        if args.aug_test == False:
            val_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='val', mode='val', **data_kwargs)
        else:
            val_dataset = get_segmentation_dataset(cfg.DATASET.NAME+"_aug", split='train', mode='aug_val', **data_kwargs)

        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=cfg.TEST.BATCH_SIZE, drop_last=False)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=1,
                                          pin_memory=True)
        self.classes = val_dataset.classes
        logging.info(self.classes)
        # create network
        self.model = get_segmentation_model().to(self.device)

        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'named_modules') and \
            cfg.MODEL.BN_EPS_FOR_ENCODER:
            logging.info('set bn custom eps for bn in encoder: {}'.format(cfg.MODEL.BN_EPS_FOR_ENCODER))
            self.set_batch_norm_attr(self.model.encoder.named_modules(), 'eps', cfg.MODEL.BN_EPS_FOR_ENCODER)

        if cfg.SOLVER.USE_FP16 == True:
            import apex
            self.model = apex.amp.initialize(self.model.cuda(), opt_level="O1")
            logging.info("Initializing mixed precision done.")

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        self.model.to(self.device)
        self.metric = SegmentationMetric(val_dataset.num_class, args.distributed)

    def set_batch_norm_attr(self, named_modules, attr, value):
        for m in named_modules:
            if isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                setattr(m[1], attr, value)

    def init(self):
        self.metric.reset()
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        logging.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        return model

    def eval(self):
        model = self.init()
        time_start = time.time()
        targets, outputs = [], []
        original_size = 256
        for i, (image, target) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)
            targets.append(target.cpu().data.numpy())
            with torch.no_grad():
                output = model.evaluate(image)
                output = F.interpolate(output, size=(original_size, original_size), mode='bilinear', align_corners=True)
                predict = torch.argmax(output[0], 1)
                predict = predict.cpu().data.numpy()
                outputs.append(predict)
            self.metric.update(output, target)
            pixAcc, mIoU = self.metric.get()
            logging.info("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
                i + 1, pixAcc * 100, mIoU * 100))
        logging.info('Eval use time: {:.3f} second'.format(time.time() - time_start))
        synchronize()

    def aug_eval(self):
        def evaluate(image, model, original_size, output_size, dims=None):
            image = F.interpolate(image, size=(output_size, output_size), mode='bilinear', align_corners=True)
            output = model.forward_8_14_to_14_v2(image)[0]
            if dims != None:
                output = torch.flip(output, dims=dims)
            output = F.interpolate(output, size=(original_size, original_size), mode='bilinear', align_corners=True)
            return output

        model = self.init()
        time_start = time.time()
        outputs, filenames = [], []
        for i, (image0, image1, image2, image3, target, filename) in enumerate(self.val_loader):
            image0 = image0.to(self.device)
            image1 = image1.to(self.device)
            image2 = image2.to(self.device)
            image3 = image3.to(self.device)
            target = target.to(self.device)
            original_size = 256
            base_size = cfg.TRAIN.BASE_SIZE
            with torch.no_grad():
                scales = cfg.TEST.SCALES
                for index, scale in enumerate(scales):
                    output_size = int(base_size * scale)
                    output0 = evaluate(image0, model, original_size, output_size)
                    output1 = evaluate(image1, model, original_size, output_size, dims=[3])
                    output2 = evaluate(image2, model, original_size, output_size, dims=[2])
                    output3 = evaluate(image3, model, original_size, output_size, dims=[2, 3])
                    if index == 0:
                        output = (output0 + output1 + output2 + output3)
                    else:
                        output += (output0 + output1 + output2 + output3)
            output = output / 12.0
            predict = torch.argmax(output, 1)
            predict = predict.cpu().data.numpy().astype(np.uint8)
            predict_ = [predict[i,:,:] for i in range(predict.shape[0])]

            for j in range(len(predict_)):
                filename_j = filename[j].replace("tif", "png")
                predict_j = predict_[j]
                save_path = os.path.join(out_path, filename_j)
                cv2.imwrite(save_path, predict_j)
            
            logging.info("Sample: {:d}".format(i + 1))
        logging.info('Eval use time: {:.3f} second'.format(time.time() - time_start))

        synchronize()

    def result_metric(self):
        pixAcc, mIoU, category_iou = self.metric.get(return_category_iou=True)
        logging.info('End validation pixAcc: {:.3f}, mIoU: {:.3f}'.format(
                pixAcc * 100, mIoU * 100))

        headers = ['class id', 'class name', 'iou']
        table = []
        for i, cls_name in enumerate(self.classes):
            table.append([cls_name, category_iou[i]])
        logging.info('Category iou: \n {}'.format(tabulate(table, headers, tablefmt='grid', showindex="always",
                                                           numalign='center', stralign='center')))

        if len(self.classes) == 8:
            weights = [0.1075814, 0.06220228, 0.18560777, 0.1683705,
                       0.08614201, 0.15660633, 0.05183542, 0.18680766]
        else:
            weights = [0.02754302, 0.07850080, 0.12062946, 0.06670495,
                       0.00267682, 0.22780647, 0.01259433, 0.12597127,
                       0.02395271, 0.15588569, 0.07321313, 0.04918330,
                       0.01127538, 0.02406267]
        weighted_iou = [weights[i]*category_iou[i] for i in range(len(weights))]
        fwiou = sum(weighted_iou)
        logging.info("FWIoU: %.6f" % (fwiou*100))


if __name__ == '__main__':
    args = parse_args()
    cfg.DIR_NAME = args.config_file.split("/")[-1].split(".")[0]
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()
    default_setup(args)

    train_id = args.config_file.split("_")[0][-4:]
    out_path = 'datasets/naicrs/datasetA/trainval/masks_' + train_id
    try:
        os.mkdir(out_path)
    except:
        pass

    evaluator = Evaluator(args)
    if args.aug_test == False:
        evaluator.eval()
    else:
        evaluator.aug_eval()
    evaluator.result_metric()
