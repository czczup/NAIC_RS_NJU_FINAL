# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import torch
import torch.nn as nn
from .layers import MyModule, ZeroLayer, set_layer_from_config
from .utils import MyNetwork


class MobileInvertedResidualBlock(MyModule):
    
    def __init__(self, mobile_inverted_conv, shortcut, ibn=False):
        super(MobileInvertedResidualBlock, self).__init__()
        
        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut
        if ibn and self.mobile_inverted_conv.config['out_channels'] <= 32: # 32/48/136/192
            self.IN = nn.InstanceNorm2d(self.mobile_inverted_conv.config['out_channels'])
        else:
            self.IN = None
    
    def forward(self, x):
        if self.mobile_inverted_conv is None or isinstance(self.mobile_inverted_conv, ZeroLayer):
            res = x
        elif self.shortcut is None or isinstance(self.shortcut, ZeroLayer):
            res = self.mobile_inverted_conv(x)
        else:
            res = self.mobile_inverted_conv(x) + self.shortcut(x)
        if self.IN is not None:
            res = self.IN(res)
        return res
    
    @property
    def module_str(self):
        return '(%s, %s)' % (
            self.mobile_inverted_conv.module_str if self.mobile_inverted_conv is not None else None,
            self.shortcut.module_str if self.shortcut is not None else None
        )
    
    @property
    def config(self):
        return {
            'name': MobileInvertedResidualBlock.__name__,
            'mobile_inverted_conv': self.mobile_inverted_conv.config if self.mobile_inverted_conv is not None else None,
            'shortcut': self.shortcut.config if self.shortcut is not None else None,
        }
    
    @staticmethod
    def build_from_config(config, ibn=False):
        mobile_inverted_conv = set_layer_from_config(config['mobile_inverted_conv'])
        shortcut = set_layer_from_config(config['shortcut'])
        return MobileInvertedResidualBlock(mobile_inverted_conv, shortcut, ibn=ibn)


class MobileNetV3(MyNetwork):

    def __init__(self, first_conv, blocks):
        super(MobileNetV3, self).__init__()

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        outs = list()
        x = self.first_conv(x)
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index == 3 or index == 7 or index == 15:
                outs.append(x)
        outs.append(x)
        return outs

    @staticmethod
    def build_from_config(config, ibn=False):
        first_conv = set_layer_from_config(config['first_conv'])
        
        blocks = []
        for block_config in config['blocks']:
            blocks.append(MobileInvertedResidualBlock.build_from_config(block_config, ibn=ibn))

        net = MobileNetV3(first_conv, blocks)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)

        return net

