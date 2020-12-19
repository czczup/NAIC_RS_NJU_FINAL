import os
import argparse
import numpy as np
from mindspore import Tensor, Parameter
import mindspore.common.dtype as mstype
import mindspore.ops as P
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.nets import net_factory
from PIL import Image
import cv2
from metrics import SegmentationMetric
import time
import warnings
warnings.filterwarnings("ignore")
context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False, device_id=0)

def parse_args():
    parser = argparse.ArgumentParser('mindspore deeplabv3plus eval')

    # val data
    parser.add_argument('--data_root', type=str, default="datasets/naicrs/final", help='root path of val data')
    parser.add_argument('--data_lst', type=str, default="datasets/naicrs/txt/final.txt", help='list of val data')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--crop_size', type=int, default=256, help='crop size')
    parser.add_argument('--image_mean', type=list, default=[0.485, 0.456, 0.406], help='image mean')
    parser.add_argument('--image_std', type=list, default=[0.229, 0.224, 0.225], help='image std')
    parser.add_argument('--scales', type=list, default=[1.0], help='scales of evaluation')
    parser.add_argument('--flip', action='store_true', help='perform left-right flip')
    parser.add_argument('--ignore_label', type=int, default=-1, help='ignore label')
    parser.add_argument('--num_classes', type=int, default=14, help='number of classes')

    parser.add_argument('--backbone', type=str, default='resnext101', help='select backbone')
    parser.add_argument('--mode', type=str, default='06', help='mode to evaluate')

    args, _ = parser.parse_known_args()
    return args


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
    batch_img = Tensor(batch_img, mstype.float32)
    net_out_0 = eval_net[0](batch_img)
    net_out_1 = eval_net[1](batch_img)
    net_out = P.TensorAdd()(net_out_0, net_out_1)
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
        result = i.argmax(axis=2)
        result[result>=3] += 1
        result_msk.append(result)
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

def to_float32(param_dict):
    new_param_dict = {}
    for k, v in param_dict.items():
        new_param_dict[k] = Parameter(Tensor(v, dtype=mstype.float32), name=k)
    return new_param_dict

def net_eval():
    write_path = "datasets/naicrs/final/results"
    if not os.path.exists(write_path):
        os.mkdir(write_path)
    
    with open(args.data_lst) as f:
        img_lst = f.readlines()
    backbone = net_factory.backbones_map[args.backbone]
    network1 = net_factory.nets_map["deeplabv3plusv2"]('eval', args.num_classes, aux=False, mode=args.mode, get_backbone=backbone)
    network2 = net_factory.nets_map["dunetv2"]('eval', args.num_classes, aux=False, mode=args.mode, get_backbone=backbone)

    eval_net = [network1, network2]
    
    """ load from .ckpt file """
    param_dict = load_checkpoint("checkpoints/resnext101_deeplabv3plus.ckpt")
    param_dict = to_float32(param_dict)
    load_param_into_net(eval_net[0], param_dict)
    eval_net[0].set_train(False)
    
    param_dict = load_checkpoint("checkpoints/resnext101_dunet.ckpt")
    param_dict = to_float32(param_dict)
    load_param_into_net(eval_net[1], param_dict)
    eval_net[1].set_train(False)

    batch_img_lst = []
    batch_msk_lst = []
    batch_name_lst = []
    bi = 0
    image_num = 0
    for i, line in enumerate(img_lst):
        img_path, msk_path = line.strip().split(' ')
        img_path = os.path.join(args.data_root, img_path)
        msk_path = os.path.join(args.data_root, msk_path)
        basename = os.path.basename(msk_path)
        img_ = Image.open(img_path)
        # img_ = img_.resize((224, 224), Image.BILINEAR)
        msk_ = Image.open(msk_path)
        # msk_ = msk_.resize((224, 224), Image.NEAREST)
        img_ = np.array(img_)
        msk_ = np.array(msk_, dtype=np.int32)
        if "datasetA" in msk_path:
            msk_ = msk_ // 100 - 1
        elif "datasetC" in msk_path or "final" in msk_path:
            # msk_[msk_==4] = 2
            msk_[msk_>=7] -= 2
            msk_ = msk_- 1
            if args.mode in ["01", "02", "03"]:
                msk_ = transform_mask(msk_)
        batch_img_lst.append(img_)
        batch_msk_lst.append(msk_)
        batch_name_lst.append(basename)
        bi += 1
        if bi == args.batch_size:
            batch_res = eval_batch_scales(args, eval_net, batch_img_lst, scales=args.scales,
                                          base_crop_size=args.crop_size)
            metric.addBatch(np.array(batch_res), np.array(batch_msk_lst))
            for index, res in enumerate(batch_res):
                res[res>=4] += 2
                res += 1
                cv2.imwrite(os.path.join(write_path, batch_name_lst[index].replace("tif", "png")),
                            res.astype(np.uint8))
            bi = 0
            batch_img_lst = []
            batch_msk_lst = []
            batch_name_lst = []
            print('processed {} images'.format(i+1))
        image_num = i

    if bi > 0:
        batch_res = eval_batch_scales(args, eval_net, batch_img_lst, scales=args.scales,
                                      base_crop_size=args.crop_size)
        metric.addBatch(np.array(batch_res), np.array(batch_msk_lst))
        for index, res in enumerate(batch_res):
            res[res >= 4] += 2
            res += 1
            cv2.imwrite(os.path.join(write_path, batch_name_lst[index].replace("tif", "png")),
                        res.astype(np.uint8))
        print('processed {} images'.format(image_num + 1))
        
    print('per-class IoU', metric.intersectionOverUnion())
    print('mIoU', metric.meanIntersectionOverUnion())
    print('FWIoU', metric.Frequency_Weighted_Intersection_over_Union())


if __name__ == '__main__':
    args = parse_args()
    print(args)
    metric = SegmentationMetric(numClass=args.num_classes+1)
    time_start = time.time()
    net_eval()
    print("using time: %.2fs" %(time.time() - time_start))
