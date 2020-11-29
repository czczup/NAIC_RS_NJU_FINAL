from __future__ import print_function

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
import time

from torchvision import transforms
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.score import SegmentationMetric
from segmentron.utils.distributed import synchronize, make_data_sampler, make_batch_data_sampler
from segmentron.config import cfg
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from tqdm import tqdm
import numpy as np
import cv2


class Tester(object):
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
            test_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='test', mode='test', **data_kwargs)
        else:
            test_dataset = get_segmentation_dataset(cfg.DATASET.NAME+"_aug_B", split='test', mode='aug_test', **data_kwargs)
        test_sampler = make_data_sampler(test_dataset, False, args.distributed)
        test_batch_sampler = make_batch_data_sampler(test_sampler, images_per_batch=cfg.TEST.BATCH_SIZE, drop_last=False)
        self.test_loader = data.DataLoader(dataset=test_dataset,
                                          batch_sampler=test_batch_sampler,
                                          num_workers=cfg.DATASET.WORKERS,
                                          pin_memory=True)
        self.classes = test_dataset.classes

        # create network
        self.model = get_segmentation_model().to(self.device)
        # self.model.load_state_dict(torch.load(test_model_path, map_location=lambda storage, loc: storage), strict=False)


        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'named_modules') and \
            cfg.MODEL.BN_EPS_FOR_ENCODER:
            logging.info('set bn custom eps for bn in encoder: {}'.format(cfg.MODEL.BN_EPS_FOR_ENCODER))
            self.set_batch_norm_attr(self.model.encoder.named_modules(), 'eps', cfg.MODEL.BN_EPS_FOR_ENCODER)

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        self.model.to(self.device)

        self.metric = SegmentationMetric(test_dataset.num_class, args.distributed)

    def set_batch_norm_attr(self, named_modules, attr, value):
        for m in named_modules:
            if isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                setattr(m[1], attr, value)

    def init(self):
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        logging.info("Start test, Total sample: {:d}".format(len(self.test_loader)))
        return model

    def test(self):
        model = self.init()
        predicts, filenames = [], []
        original_size = 256
        for i, (image, filename) in enumerate(tqdm(self.test_loader)):
            image = image.to(self.device)
            with torch.no_grad():
                output = model(image)
                output = F.interpolate(output, size=(original_size, original_size), mode='bilinear', align_corners=False)
                predict = torch.argmax(output[0], 1)
                predict = predict.cpu().data.numpy()
                predict = ((predict + 1) * 100).astype(np.uint16)
                predicts += [item for item in predict]
                filenames += filename
        for i in range(len(predicts)):
            filename = filenames[i].replace("tif", "png")
            predict = predicts[i]
            cv2.imwrite(os.path.join(args.out, filename), predict)
        synchronize()

    def aug_test(self):
        def evaluate(image, model, original_size, output_size, dims=None):
            image = F.interpolate(image, size=(output_size, output_size), mode='bilinear', align_corners=False)
            output = model.evaluate(image)
            if dims != None:
                output = torch.flip(output, dims=dims)
            output = F.interpolate(output, size=(original_size, original_size), mode='bilinear', align_corners=False)
            output = F.softmax(output, dim=1) # (batch, channel, width, height)
            return output

        model = self.init()
        predicts, filenames, outputs = [], [], []
        original_size = 256
        base_size = cfg.TRAIN.BASE_SIZE
        scales = cfg.TEST.SCALES
        for i, (image0, image1, image2, image3, filename) in enumerate(tqdm(self.test_loader)):
            image0 = image0.to(self.device)
            image1 = image1.to(self.device)
            image2 = image2.to(self.device)
            image3 = image3.to(self.device)
            with torch.no_grad():
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

                output_ = (output.cpu().numpy()).astype(np.float16)
                outputs += [output_[i, :, :] for i in range(output_.shape[0])]

                predict = torch.argmax(output, 1)
                predict = predict.cpu().data.numpy()
                predict = ((predict + 1) * 100).astype(np.uint16)

                predicts += [predict[i,:,:] for i in range(predict.shape[0])]
                filenames += filename

        train_id = args.config_file.split("_")[0][-4:]
        try:
            if not os.path.exists(args.out):
                os.mkdir(args.out)
            if not os.path.exists(args.out + '_' + train_id):
                os.mkdir(args.out + '_' + train_id)
        except:
            pass

        for i in range(len(predicts)):
            filename = filenames[i].replace("tif", "png")
            predict = predicts[i]
            cv2.imwrite(os.path.join(args.out, filename), predict)

        for i in range(len(outputs)):
            filename = filenames[i].replace("tif", "npy")
            output = outputs[i]
            np.save(os.path.join(args.out+'_'+train_id, filename), output)
        synchronize()


if __name__ == '__main__':
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()
    default_setup(args)

    torch.backends.cudnn.benchmark = True

    tester = Tester(args)
    if args.aug_test == False:
        tester.test()
    else:
        tester.aug_test()
    torch.cuda.empty_cache()