import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(os.path.split(cur_path)[0])[0]
sys.path.append(root_path)

from mindspore import context
from mindspore.train.serialization import save_checkpoint, load_checkpoint, load_param_into_net
from src.nets.backbones.resnext import resnext50_32x4d as ms_resnext50
from src.nets.backbones.resnext import resnext101_32x8d as ms_resnext101
from segmentron.models.backbones.resnext import resnext50_32x4d as pt_resnext50
from segmentron.models.backbones.resnext import resnext101_32x8d as pt_resnext101
from src.nets.deeplabv3plusv2 import DeepLabV3PlusV2 as Ms_DeepLabV3PlusV2
from src.nets.pytorch_deeplabv3plus import DeepLabV3PlusNearest as Pt_DeepLabV3PlusV2
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
    
    key = key.replace("aspp0.conv", "aspp0.0")
    key = key.replace("aspp0.bn.weight", "aspp0.1.gamma")
    key = key.replace("aspp0.bn.bias", "aspp0.1.beta")
    key = key.replace("aspp0.bn", "aspp0.1")
    
    key = key.replace("bn_depth.weight", "bn_depth.gamma")
    key = key.replace("bn_depth.bias", "bn_depth.beta")
    key = key.replace("bn_point.weight", "bn_point.gamma")
    key = key.replace("bn_point.bias", "bn_point.beta")
    
    key = key.replace("image_pooling.conv", "image_pooling.0")
    
    key = key.replace("image_pooling.bn.weight", "image_pooling.1.gamma")
    key = key.replace("image_pooling.bn.bias", "image_pooling.1.beta")
    key = key.replace("image_pooling.bn", "image_pooling.1")
    
    key = key.replace("bn.weight", "bn.gamma")
    key = key.replace("bn.bias", "bn.beta")
    
    key = key.replace("auxlayer.block.1.weight", "auxlayer.block.1.gamma")
    key = key.replace("auxlayer.block.1.bias", "auxlayer.block.1.beta")
    
    key = key.replace("auxlayer2.block.1.weight", "auxlayer2.block.1.gamma")
    key = key.replace("auxlayer2.block.1.bias", "auxlayer2.block.1.beta")
    
    return key


if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False, device_id=0)
    ms_model = Ms_DeepLabV3PlusV2(phase="val", num_classes=[8, 14], output_stride=8, aux=False, freeze_bn=True,
                                  mode="03", get_backbone=ms_resnext101)
    # save_checkpoint(ms_model, "resnext101_deeplabv3plus.ckpt")
    # param_dict = load_checkpoint("resnext101_deeplabv3plus.ckpt")
    # ms_keys = list(param_dict.keys())
    # print(len(ms_keys))
    
    pt_model = Pt_DeepLabV3PlusV2(nclass=[8, 14], get_backbone=pt_resnext101, channels=[256, 2048])
    state_dict = torch.load("resnext101_deeplabv3plus.pth")
    pt_model.load_state_dict(state_dict=state_dict, strict=False)
    pt_model.eval()
    state_dict = {k: v for k, v in state_dict.items() if "num_batches_tracked" not in k and "aux" not in k}
    pt_keys = list(state_dict.keys())

    param_dict = dict()
    for k, v in state_dict.items():
        parameter = v.cpu().data.numpy()
        parameter = Parameter(Tensor(parameter), name=key_mapping(k))
        param_dict[key_mapping(k)] = parameter
        
    load_param_into_net(ms_model, param_dict)
    save_checkpoint(ms_model, "resnext101_deeplabv3plus.ckpt")
    ms_model.set_train(False)
    
    
    image = Tensor(np.ones((1, 3, 256, 256), dtype=np.float32), mstype.float32)
    net_out = ms_model(image)
    net_out = net_out.asnumpy()[0][0]
    print(net_out)
    print(net_out.shape)

    image = torch.ones(1, 3, 256, 256)
    out = pt_model(image)[0][0]
    print(out)
    print(out.shape)
    exit(0)
