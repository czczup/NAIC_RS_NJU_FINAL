import json
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import os
from .model_zoo import MobileNetV3
from segmentron.models.backbones import BACKBONE_REGISTRY

__all__ = ['ofa_note10_lat_8ms', 'ibn_ofa_note10_lat_8ms']

@BACKBONE_REGISTRY.register()
def ofa_note10_lat_8ms(norm_layer=nn.BatchNorm2d):
    net_config = json.load(open(os.path.join(os.path.dirname(__file__), 'net.config'), 'r'))
    return MobileNetV3.build_from_config(net_config)

@BACKBONE_REGISTRY.register()
def ibn_ofa_note10_lat_8ms(norm_layer=nn.BatchNorm2d):
    net_config = json.load(open(os.path.join(os.path.dirname(__file__), 'net.config'), 'r'))
    return MobileNetV3.build_from_config(net_config, ibn=True)

if __name__ == '__main__':
    net_config = json.load(open('net.config', 'r'))
    net = MobileNetV3.build_from_config(net_config)
    device = torch.device("cuda")
    dummy_input = torch.randn([1, 3, 256, 256]).to(device)
    net = net.to(device)
    outs = net(dummy_input)
    for out in outs:
        print(out.shape)
    torch.cuda.synchronize()
    time_start = time.time()
    for i in tqdm(range(1000)):
        outs = net(dummy_input)
    torch.cuda.synchronize()
    print(time.time()-time_start)
    # print(net)