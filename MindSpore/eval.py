# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""eval deeplabv3."""

import os
import argparse
import numpy as np
import cv2
from mindspore import Tensor, Parameter
import mindspore.common.dtype as mstype
import mindspore.ops as P
import mindspore.nn as nn
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.nets import net_factory
import torch
context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False,
                    device_id=int(os.getenv('DEVICE_ID')))


def parse_args():
    parser = argparse.ArgumentParser('mindspore deeplabv3plus eval')

    # val data
    parser.add_argument('--data_root', type=str, default='', help='root path of val data')
    parser.add_argument('--data_lst', type=str, default='', help='list of val data')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--crop_size', type=int, default=256, help='crop size')
    parser.add_argument('--image_mean', type=list, default=[0.485, 0.456, 0.406], help='image mean')
    parser.add_argument('--image_std', type=list, default=[0.229, 0.224, 0.225], help='image std')
    parser.add_argument('--scales', type=float, action='append', help='scales of evaluation')
    parser.add_argument('--flip', action='store_true', help='perform left-right flip')
    parser.add_argument('--ignore_label', type=int, default=-1, help='ignore label')
    parser.add_argument('--num_classes', type=int, default=14, help='number of classes')

    # model
    parser.add_argument('--model', type=str, default='deeplab_v3_s16', help='select model')
    parser.add_argument('--freeze_bn', action='store_true', default=False, help='freeze bn')
    parser.add_argument('--pth_path', type=str, default='', help='model to evaluate')
    parser.add_argument('--ckpt_path', type=str, default='', help='model to evaluate')

    args, _ = parser.parse_known_args()
    return args


def cal_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(np.int32) + b[k], minlength=n ** 2).reshape(n, n)


def resize_long(img, long_size=513):
    h, w, _ = img.shape
    if h > w:
        new_h = long_size
        new_w = int(1.0 * long_size * w / h)
    else:
        new_w = long_size
        new_h = int(1.0 * long_size * h / w)
    imo = cv2.resize(img, (new_w, new_h))
    return imo



def pre_process(args, img_, crop_size=513):
    # resize
    img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)  # to RGB image
    img_ = resize_long(img_, crop_size)
    resize_h, resize_w, _ = img_.shape

    # mean, std
    image_mean = np.array(args.image_mean)
    image_std = np.array(args.image_std)
    img_ = img_ / 255.0
    img_ = (img_ - image_mean) / image_std

    # pad to crop_size
    pad_h = crop_size - img_.shape[0]
    pad_w = crop_size - img_.shape[1]
    if pad_h > 0 or pad_w > 0:
        img_ = cv2.copyMakeBorder(img_, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    # hwc to chw
    img_ = img_.transpose((2, 0, 1))
    return img_, resize_h, resize_w


def eval_batch(args, eval_net, img_lst, crop_size=256):
    result_lst = []
    batch_size = len(img_lst)
    batch_img = np.zeros((args.batch_size, 3, crop_size, crop_size), dtype=np.float32)
    resize_hw = []
    for l in range(batch_size):
        img_ = img_lst[l]
        img_, resize_h, resize_w = pre_process(args, img_, crop_size)
        batch_img[l] = img_
        resize_hw.append([resize_h, resize_w])

    batch_img = np.ascontiguousarray(batch_img)
    net_out = eval_net(Tensor(batch_img, mstype.float32))
    net_out = net_out.asnumpy()

    for bs in range(batch_size):
        probs_ = net_out[bs][:, :resize_hw[bs][0], :resize_hw[bs][1]].transpose((1, 2, 0))
        ori_h, ori_w = img_lst[bs].shape[0], img_lst[bs].shape[1]
        probs_ = cv2.resize(probs_, (ori_w, ori_h))
        result_lst.append(probs_)

    return result_lst


def eval_batch_scales(args, eval_net, img_lst, scales,
                      base_crop_size=256):
    sizes_ = [int((base_crop_size - 1) * sc) + 1 for sc in scales]
    probs_lst = eval_batch(args, eval_net, img_lst, crop_size=sizes_[0])
    for crop_size_ in sizes_[1:]:
        probs_lst_tmp = eval_batch(args, eval_net, img_lst, crop_size=crop_size_)
        for pl, _ in enumerate(probs_lst):
            probs_lst[pl] += probs_lst_tmp[pl]

    result_msk = []
    for i in probs_lst:
        result_msk.append(i.argmax(axis=2))
    return result_msk


def net_eval():
    args = parse_args()

    # data list
    with open(args.data_lst) as f:
        img_lst = f.readlines()

    # network
    print(args.freeze_bn)
    network = net_factory.nets_map[args.model]('eval', args.num_classes, 8, False, args.freeze_bn)

    eval_net = network

    p2m = open("runs/checkpoints/pytorch2mindspore.csv", "r+")
    p2m = [line[:-1].split(",") for line in p2m.readlines()]
    p2m = {item[0]:item[1] for item in p2m}

    state_dict = torch.load(args.pth_path)
    param_dict = dict()
    for k, v in state_dict.items():
        print(k, k in p2m)
        if k in p2m:
            parameter = v.cpu().data.numpy()
            print(len(parameter.shape))
            # if len(parameter.shape) == 4:
            #     parameter = parameter.transpose(1, 0, 2, 3)
            parameter = Parameter(Tensor(parameter), name=p2m[k])
            param_dict[p2m[k]] = parameter
    load_param_into_net(eval_net, param_dict)
    eval_net.set_train(False)
    # debug
    # c1 = Tensor(-np.ones([1, 32, 160, 160]), mstype.float32)
    # c4 = Tensor(-np.ones([1, 248, 80, 80]), mstype.float32)
    # net_out = eval_net(c4, c1)
    # net_out = net_out.asnumpy()[0][-2]
    # net_out = net_out.asnumpy()
    # print(net_out.sum(3).sum(2))
    
    # print(net_out)
    # print(net_out[...,37:43,37:43])
    # print(net_out.shape)
    # print(param_dict['network.head.aspp.image_pooling.0.weight'][0])
    # print(param_dict['network.head.aspp.image_pooling.1.gamma'][0])
    # print(param_dict['network.head.aspp.image_pooling.1.beta'][0])
    # print(param_dict['network.head.aspp.image_pooling.1.moving_mean'][0])
    # print(param_dict['network.head.aspp.image_pooling.1.moving_variance'][0])
    # exit(0)
    # evaluate
    hist = np.zeros((args.num_classes, args.num_classes))
    batch_img_lst = []
    batch_msk_lst = []
    bi = 0
    image_num = 0
    for i, line in enumerate(img_lst):
        img_path, msk_path = line.strip().split(' ')
        img_path = os.path.join(args.data_root, img_path)
        msk_path = os.path.join(args.data_root, msk_path)
        basename = os.path.basename(msk_path)
        img_ = cv2.imread(img_path)
        msk_ = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        if "datasetA" in msk_path:
            msk_ = msk_ // 100 - 1
        elif "datasetC" in msk_path:
            msk_[msk_==4] = 2
            msk_[msk_>=7] -= 3
            msk_ = msk_- 1
        batch_img_lst.append(img_)
        batch_msk_lst.append(msk_)
        bi += 1
        if bi == args.batch_size:
            batch_res = eval_batch_scales(args, eval_net, batch_img_lst, scales=args.scales,
                                          base_crop_size=args.crop_size)
            res = batch_res[0]
            cv2.imwrite(basename.split(".")[0] + "_.png", res)

            for mi in range(args.batch_size):
                hist += cal_hist(batch_msk_lst[mi].flatten(), batch_res[mi].flatten(), args.num_classes)

            bi = 0
            batch_img_lst = []
            batch_msk_lst = []
            print('processed {} images'.format(i+1))
        image_num = i

    if bi > 0:
        batch_res = eval_batch_scales(args, eval_net, batch_img_lst, scales=args.scales,
                                      base_crop_size=args.crop_size)
        res = batch_res[0]
        cv2.imwrite(basename.split(".")[0]+"_.png", res)
        for mi in range(bi):
            hist += cal_hist(batch_msk_lst[mi].flatten(), batch_res[mi].flatten(), args.num_classes)
        print('processed {} images'.format(image_num + 1))

    # print(hist)
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('per-class IoU', iu)
    print('mean IoU', np.nanmean(iu))


if __name__ == '__main__':
    print('main!')
    net_eval()
