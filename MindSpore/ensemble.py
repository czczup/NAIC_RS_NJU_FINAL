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
from mindspore import Tensor, Parameter
import mindspore.common.dtype as mstype
import mindspore.ops as P
import mindspore.nn as nn
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.nets import net_factory
from PIL import Image
from mindspore.train.serialization import save_checkpoint
try:
    import torch
except:
    pass

device = "GPU"
context.set_context(mode=context.GRAPH_MODE, device_target=device, save_graphs=False,
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
    parser.add_argument('--scales', type=list, default=[1.0], help='scales of evaluation')
    parser.add_argument('--flip', action='store_true', help='perform left-right flip')
    parser.add_argument('--ignore_label', type=int, default=-1, help='ignore label')
    parser.add_argument('--num_classes', type=int, default=8, help='number of classes')

    # model
    parser.add_argument('--model', type=str, default='deeplabv3plusv2', help='select model')
    parser.add_argument('--backbone', type=str, default='resnext101', help='select backbone')
    parser.add_argument('--pth_path', type=str, default='', help='model to evaluate')
    parser.add_argument('--ckpt_path', type=str, default='', help='model to evaluate')
    parser.add_argument('--mode', type=str, default='03', help='mode to evaluate')

    args, _ = parser.parse_known_args()
    return args


def cal_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(np.int32) + b[k], minlength=n ** 2).reshape(n, n)


def pre_process(args, img_):
    # mean, std
    image_mean = np.array(args.image_mean)
    image_std = np.array(args.image_std)
    img_ = img_ / 255.0
    img_ = (img_ - image_mean) / image_std
    # hwc to chw
    img_ = img_.transpose((2, 0, 1))
    return img_


def eval_batch(args, eval_net, img_lst, crop_size=256):
    result_lst = []
    batch_size = len(img_lst)
    batch_img = np.zeros((args.batch_size, 3, crop_size, crop_size), dtype=np.float32)
    for l in range(batch_size):
        img_ = img_lst[l]
        img_ = pre_process(args, img_)
        batch_img[l] = img_

    batch_img = np.ascontiguousarray(batch_img)
    net_out = eval_net(Tensor(batch_img, mstype.float32))
    net_out = net_out.asnumpy()
    for bs in range(batch_size):
        probs_ = net_out[bs].transpose((1, 2, 0))
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


def transform_mask(mask):
    mask[mask == 0] = 100  # 水体   -> 水体
    mask[mask == 1] = 200  # 道路   -> 道路
    mask[mask == 2] = 300  # 建筑物   -> 建筑物
    mask[mask == 3] = 800  # 停车场   -> 其它
    mask[mask == 4] = 800  # 操场    -> 其它
    mask[mask == 5] = 400  # 普通耕地 -> 耕地
    mask[mask == 6] = 400  # 农业大棚 -> 耕地
    mask[mask == 7] = 500  # 自然草地 -> 草地
    mask[mask == 8] = 500  # 绿地绿化 -> 草地
    mask[mask == 9] = 600  # 自然林   -> 林地
    mask[mask == 10] = 600  # 人工林   -> 林地
    mask[mask == 11] = 700  # 自然裸土 -> 裸土
    mask[mask == 12] = 700  # 人为裸土 -> 裸土
    mask[mask == 13] = 800  # 其它    -> 其它
    mask = mask // 100 - 1
    return mask


def net_eval():
    args = parse_args()
    print(args)
    with open(args.data_lst) as f:
        img_lst = f.readlines()
    backbone = net_factory.backbones_map[args.backbone]
    network = net_factory.nets_map[args.model]('eval', args.num_classes, aux=False, mode=args.mode, get_backbone=backbone)
    eval_net = network
    
    """ load from .ckpt file """
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(eval_net, param_dict)
    eval_net.set_train(False)

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
        img_ = Image.open(img_path)
        # img_ = img_.resize((args.crop_size, args.crop_size), Image.BILINEAR)
        msk_ = Image.open(msk_path)
        # msk_ = msk_.resize((args.crop_size, args.crop_size), Image.NEAREST)
        img_ = np.array(img_)
        msk_ = np.array(msk_, dtype=np.int32)
        if "datasetA" in msk_path:
            msk_ = msk_ // 100 - 1
        elif "datasetC" in msk_path:
            msk_[msk_==4] = 2
            msk_[msk_>=7] -= 3
            msk_ = msk_- 1
            if args.mode in ["01", "02", "03"]:
                msk_ = transform_mask(msk_)
        batch_img_lst.append(img_)
        batch_msk_lst.append(msk_)
        bi += 1
        if bi == args.batch_size:
            batch_res = eval_batch_scales(args, eval_net, batch_img_lst, scales=args.scales,
                                          base_crop_size=args.crop_size)

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
        for mi in range(bi):
            hist += cal_hist(batch_msk_lst[mi].flatten(), batch_res[mi].flatten(), args.num_classes)
        print('processed {} images'.format(image_num + 1))

    # print(hist)
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('per-class IoU', iu)
    print('mean IoU', np.nanmean(iu))


if __name__ == '__main__':
    net_eval()
