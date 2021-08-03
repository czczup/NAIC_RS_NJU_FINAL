import json
import torch.utils.data
from .model_zoo import ProxylessNASNets
import time
from tqdm import tqdm
from segmentron.models.backbones import BACKBONE_REGISTRY
import torch.nn as nn
import os

__all__ = ['ofa_1080ti_gpu64_27ms']


@BACKBONE_REGISTRY.register()
def ofa_1080ti_gpu64_27ms(norm_layer=nn.BatchNorm2d):
    net_config = json.load(open(os.path.join(os.path.dirname(__file__), 'net.config'), 'r'))
    return ProxylessNASNets.build_from_config(net_config)


if __name__ == '__main__':
    
    # build network
    net_config = json.load(open(os.path.join(os.path.dirname(__file__), 'net.config'), 'r'))
    
    net = ProxylessNASNets.build_from_config(net_config)
    
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
    print(net)