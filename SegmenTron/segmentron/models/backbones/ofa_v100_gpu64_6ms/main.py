import json
import os
import torch.utils.data
from .model_zoo import ProxylessNASNets
from segmentron.models.backbones import BACKBONE_REGISTRY
import torch.nn as nn

__all__ = ['ofa_v100_gpu64_6ms', 'ibn_ofa_v100_gpu64_6ms']

@BACKBONE_REGISTRY.register()
def ofa_v100_gpu64_6ms(norm_layer=nn.BatchNorm2d):
    net_config = json.load(open(os.path.join(os.path.dirname(__file__), 'net.config'), 'r'))
    return ProxylessNASNets.build_from_config(net_config)

@BACKBONE_REGISTRY.register()
def ibn_ofa_v100_gpu64_6ms(norm_layer=nn.BatchNorm2d):
    net_config = json.load(open(os.path.join(os.path.dirname(__file__), 'net.config'), 'r'))
    return ProxylessNASNets.build_from_config(net_config, ibn=True)


if __name__ == '__main__':
    net_config = json.load(open(os.path.join(os.path.dirname(__file__), 'net.config'), 'r'))
    net = ProxylessNASNets.build_from_config(net_config)
    device = torch.device("cuda")
    dummy_input = torch.randn([1, 3, 256, 256]).to(device)
    net = net.to(device)
    outs = net(dummy_input)
    for out in outs:
        print(out.shape)
    print(net)