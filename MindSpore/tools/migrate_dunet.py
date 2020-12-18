import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from mindspore import context
from mindspore.train.serialization import save_checkpoint, load_param_into_net, load_checkpoint
from src.nets.backbones.resnext import resnext101_32x8d as ms_resnext101
from segmentron.models.backbones.resnext import resnext101_32x8d as pt_resnext101
from src.nets.dunetv2 import DUNetV2 as Ms_DUNetV2
from src.pytorch.dunet import DUNet as Pt_DUNetV2
from mindspore import Tensor, Parameter
import mindspore.common.dtype as mstype
import torch
import numpy as np


def key_mapping(key):
    key = key.replace("bn0.weight", "bn0.gamma")
    key = key.replace("bn1.weight", "bn1.gamma")
    key = key.replace("bn2.weight", "bn2.gamma")
    key = key.replace("bn3.weight", "bn3.gamma")
    
    key = key.replace("bn0.bias", "bn0.beta")
    key = key.replace("bn1.bias", "bn1.beta")
    key = key.replace("bn2.bias", "bn2.beta")
    key = key.replace("bn3.bias", "bn3.beta")
    
    key = key.replace("running_mean", "moving_mean")
    key = key.replace("running_var", "moving_variance")
    
    key = key.replace("downsample.1.weight", "downsample.1.gamma")
    key = key.replace("downsample.1.bias", "downsample.1.beta")
    
    key = key.replace("conv2.1.weight", "conv2.1.gamma")
    key = key.replace("conv2.1.bias", "conv2.1.beta")
    key = key.replace("conv3.1.weight", "conv3.1.gamma")
    key = key.replace("conv3.1.bias", "conv3.1.beta")

    key = key.replace("block.1.weight", "block.1.gamma")
    key = key.replace("block.1.bias", "block.1.beta")

    key = key.replace("block.4.weight", "block.4.gamma")
    key = key.replace("block.4.bias", "block.4.beta")
    return key


if __name__ == '__main__':
    read_path = "checkpoints/resnext101_dunet.pth"
    write_path = read_path.replace(".pth", ".ckpt")
    
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False, device_id=0)
    
    print("load pytorch model from:", read_path)
    state_dict = torch.load(read_path)
    state_dict = {k: v for k, v in state_dict.items() if "num_batches_tracked" not in k and "aux" not in k}

    save_obj = []
    for k, v in state_dict.items():
        parameter = v.cpu().data.numpy()
        parameter = Parameter(Tensor(parameter, dtype=mstype.float16), name=key_mapping(k))
        save_obj.append({"name": key_mapping(k), "data": parameter})

    save_checkpoint(save_obj, write_path)
    print("save mindspore model to:", write_path)
