import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import os
import math
import time
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
from tqdm import tqdm

batch_size = 8
image_mean = np.array([0.485, 0.456, 0.406])
image_std = np.array([0.229, 0.224, 0.225])


class Scale():
    def __init__(self, crop_size, upsample_rate, stride):
        self.crop_size = crop_size
        self.upsample_rate = upsample_rate
        self.base_size = int(crop_size * upsample_rate)
        self.stride = stride

scale = Scale(crop_size=256, upsample_rate=1.0, stride=256-16)


def cut(image, scale):
    height, width, _ = image.shape
    row_num = (height - scale.crop_size) // scale.stride + 1
    col_num = (width - scale.crop_size) // scale.stride + 1
    image_list = np.zeros((int(row_num*col_num), 3, scale.crop_size, scale.crop_size), dtype=np.float32)
    cnt = 0
    for j in range(0, row_num):
        b = j * scale.stride
        d = b + scale.crop_size
        for i in range(0, col_num):
            a = i * scale.stride
            c = a + scale.crop_size
            image_list[cnt] = pre_process(image[b:d, a:c])
            cnt += 1
    return image_list, row_num, col_num

def restore(image_list, row_num, col_num):

    row_images = []
    blending_width = (scale.crop_size - scale.stride) // 2
    image_list = [image_list[i:i + col_num] for i in range(0, len(image_list), col_num)]
    
    for i, row in enumerate(image_list):
        row_image = None
        for j, item in enumerate(row):
            if j == 0:
                row_image = item
            else:
                if blending_width != 0:
                    left = row_image[:, 0:-2*blending_width]
                    overlap1 = row_image[:, -2*blending_width:]
                    overlap2 = item[:, 0:2*blending_width]
                    right = item[:, 2*blending_width:]
                    overlap = overlap1 + overlap2
                    row_image = cv2.hconcat([left, overlap, right])
                else:
                    row_image = cv2.hconcat([row_image, item])
        row_images.append(row_image)

    image = None
    for i, row in enumerate(row_images):
        if i == 0:
            image = row
        else:
            if blending_width != 0:
                top = image[0:-2*blending_width, :]
                overlap1 = image[-2*blending_width:, :]
                overlap2 = row[0:2*blending_width, :]
                bottom = row[2*blending_width:, :]
                overlap = overlap1 + overlap2
                image = cv2.vconcat([top, overlap, bottom])
            else:
                image = cv2.vconcat([image, row])

    return image


def pre_process(img):
    img = img / 255.0
    img = (img - image_mean) / image_std
    img = img.transpose((2, 0, 1))
    return img

def predict(model, input_path, output_dir):
    filename = os.path.basename(input_path)
    filename, _ = os.path.splitext(filename)
    image = Image.open(input_path).convert('RGB')
    
    # image = image.resize((1000, 1000)) # TODO: remove
    time_start = time.time()
    origin_width, origin_height = image.size
    pad_width = math.ceil((origin_width - scale.crop_size) / scale.stride) * scale.stride + scale.crop_size
    pad_height = math.ceil((origin_height - scale.crop_size) / scale.stride) * scale.stride + scale.crop_size
    image = image.resize((pad_width, pad_height))
    image = np.array(image)
    image_list, row_num, col_num = cut(image, scale)
    print("pre-process: %.2f" % (time.time() - time_start))
    
    time_start = time.time()
    result = single_scale_predict_v2(model, image_list)
    print("predict: %.2f" % (time.time() - time_start))

    time_start = time.time()
    result = restore(result, row_num, col_num)
    result = cv2.resize(result, (origin_width, origin_height))
    result = np.argmax(result, axis=2).astype(np.uint8)
    result += 1
    result[result >= 4] += 3
    cv2.imwrite(os.path.join(output_dir, filename+".png"), result)
    print("post-process: %.2f" % (time.time() - time_start))


def single_scale_predict_v1(model, image):
    width, height = image.size(2), image.size(3)
    image = F.interpolate(image, (int(width * scale.upsample_rate), int(height * scale.upsample_rate)),
                          mode='bilinear', align_corners=True)
    output = model(image)[0]
    return output


def single_scale_predict_v2(model, image_list):
    batch_num = math.ceil(len(image_list) / batch_size)
    outputs = []
    for i in range(batch_num):
        batch = image_list[i * batch_size:(i + 1) * batch_size]
        output = model(Tensor(batch, mstype.float32))
        output = output.asnumpy()
        for bs in range(output.shape[0]):
            probs_ = output[bs].transpose((1, 2, 0))
            outputs.append(probs_)
    return outputs


def init_model(ckpt_path):
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False, device_id=0)
    backbone = net_factory.backbones_map["resnext101"]
    model = net_factory.nets_map["deeplabv3plusv2"]('eval', [8, 14], aux=False, mode="03", get_backbone=backbone)
    print("loading checkpoint from", ckpt_path)
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(model, param_dict)
    print("eval mode")
    model.set_train(False)
    return model
    
    
if __name__ == '__main__':
    # dataset = "datasetA"
    dataset = "gid"
    ckpt_path = "runs/checkpoints/resnext101_deeplabv3plus.ckpt"
    if dataset == "datasetA":
        data_dir = "datasets/naicrs/datasetA/trainval"
        output_dir = "datasets/naicrs/datasetA/trainval/results"
        f = open("datasets/naicrs/txt/valA.txt", "r")
        filenames = [item.replace("\n", "").split(" ")[0] for item in f.readlines()]
        input_paths = [os.path.join(data_dir, filename) for filename in filenames]
    elif dataset == "gid":
        data_dir = "datasets/gid/images"
        output_dir = "datasets/gid/results"
        input_paths = [os.path.join(data_dir, str(i)+".tif") for i in range(1, 10+1)]
        
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    model = init_model(ckpt_path=ckpt_path)
    start_time = time.time()
    for input_path in tqdm(input_paths):
        predict(model, input_path, output_dir)
    print("using time: %.2fs" % (time.time() - start_time))