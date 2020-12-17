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
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False, device_id=0)
    ms_model = Ms_DUNetV2(phase="val", num_classes=[8, 14], aux=False, get_backbone=ms_resnext101, mode="03")
    # save_checkpoint(ms_model, "resnext101_dunet.ckpt")
    # param_dict = load_checkpoint("resnext101_dunet.ckpt")
    # ms_keys = list(param_dict.keys())
    
    pt_model = Pt_DUNetV2(nclass=[8, 14], get_backbone=pt_resnext101)
    state_dict = torch.load("resnext101_dunet.pth")
    pt_model.load_state_dict(state_dict=state_dict, strict=False)
    # state_dict = pt_model.state_dict()
    pt_model.eval()
    state_dict = {k: v for k, v in state_dict.items() if "num_batches_tracked" not in k and "aux" not in k}
    pt_keys = list(state_dict.keys())

    # for i in range(len(ms_keys)):
    #     if key_mapping(pt_keys[i]) not in ms_keys:
    #         print(key_mapping(pt_keys[i]), ms_keys[i])

    
    param_dict = dict()
    for k, v in state_dict.items():
        parameter = v.cpu().data.numpy()
        parameter = Parameter(Tensor(parameter), name=key_mapping(k))
        param_dict[key_mapping(k)] = parameter

    load_param_into_net(ms_model, param_dict)
    save_checkpoint(ms_model, "resnext101_dunet.ckpt")
    ms_model.set_train(False)

    image = Tensor(np.ones((1, 3, 256, 256), dtype=np.float32), mstype.float32)
    net_out = ms_model(image)
    net_out = net_out.asnumpy()[0][0]
    print(net_out)
    print(net_out.shape)

    image = torch.ones(1, 3, 256, 256)
    out = pt_model(image, "03")[0][0][0]
    print(out)
    print(out.shape)
