# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

from collections import OrderedDict
import math
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import operations as P
import mindspore
import json
import os


def build_activation(act_func):
    if act_func == 'relu':
        return nn.ReLU()
    elif act_func == 'relu6':
        return nn.ReLU6()
    elif act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'sigmoid':
        return nn.Sigmoid()
    elif act_func == 'h_swish':
        return nn.HSwish()
    elif act_func is None:
        return None
    else:
        raise ValueError('do not support: %s' % act_func)

def set_layer_from_config(layer_config):
    if layer_config is None:
        return None

    name2layer = {
        ConvLayer.__name__: ConvLayer,
        MBInvertedConvLayer.__name__: MBInvertedConvLayer,
    }

    layer_name = layer_config.pop('name')
    print(layer_name)
    if layer_name == "IdentityLayer":
        return True
    else:
        layer = name2layer[layer_name]
        return layer.build_from_config(layer_config)


class ConvLayer(nn.Cell):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=2, dilation=1, groups=1, bias=False, has_shuffle=False,
                 use_bn=True, act_func='relu6', dropout_rate=0, ops_order='weight_bn_act'):
        super(ConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.conv = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, pad_mode='same',
            dilation=self.dilation, group=self.groups, has_bias=self.bias, weight_init='he_uniform')
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.act = build_activation(act_func)

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

    @staticmethod
    def build_from_config(config):
        return ConvLayer(**config)


class MBInvertedConvLayer(nn.Cell):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, expand_ratio=6, dilation=1, mid_channels=None, act_func='relu6', use_se=False):
        super(MBInvertedConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels
        self.act_func = act_func
        self.use_se = use_se
        self.dilation = dilation
        if self.mid_channels is None:
            feature_dim = round(self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        if self.expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.SequentialCell(OrderedDict([
                ('conv', nn.Conv2d(self.in_channels, feature_dim, 1, 1, pad_mode='valid', has_bias=False, weight_init='he_uniform')),
                ('bn', nn.BatchNorm2d(feature_dim)),
                ('act', build_activation(self.act_func)),
            ]))

        depth_conv_modules = [
            ('conv', nn.Conv2d(feature_dim, feature_dim, kernel_size, stride, pad_mode='same',
                               group=feature_dim, has_bias=False, dilation=dilation, weight_init='he_uniform')),
            ('bn', nn.BatchNorm2d(feature_dim)),
            ('act', build_activation(self.act_func))
        ]
        
        self.depth_conv = nn.SequentialCell(OrderedDict(depth_conv_modules))

        self.point_linear = nn.SequentialCell(OrderedDict([
            ('conv', nn.Conv2d(feature_dim, out_channels, 1, 1, pad_mode='valid', has_bias=False, weight_init='he_uniform')),
            ('bn', nn.BatchNorm2d(out_channels)),
        ]))

    def construct(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    @staticmethod
    def build_from_config(config):
        return MBInvertedConvLayer(**config)


class MobileInvertedResidualBlock(nn.Cell):
    
    def __init__(self, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()
        
        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut
        self.add = P.TensorAdd()

    def construct(self, x):
        if self.mobile_inverted_conv is None:
            res = x
        elif self.shortcut:
            res = self.add(self.mobile_inverted_conv(x), x)
        else:
            res = self.mobile_inverted_conv(x)
        return res
    
    @staticmethod
    def build_from_config(config):
        mobile_inverted_conv = set_layer_from_config(config['mobile_inverted_conv'])
        shortcut = set_layer_from_config(config['shortcut'])
        block = MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)
        return block


class ProxylessNASNets(nn.Cell):
    
    def __init__(self, first_conv, blocks):
        super(ProxylessNASNets, self).__init__()
        
        self.first_conv = first_conv
        self.block0 = blocks[0]
        self.block1 = blocks[1]
        self.block2 = blocks[2]
        self.block3 = blocks[3]
        self.block4 = blocks[4]
        self.block5 = blocks[5]
        self.block6 = blocks[6]
        self.block7 = blocks[7]
        self.block8 = blocks[8]
        self.block9 = blocks[9]
        self.block10 = blocks[10]
        self.block11 = blocks[11]
        self.block12 = blocks[12]
        self.block13 = blocks[13]

        self.print = ops.Print()
        self.shape = ops.Shape()
        
    def construct(self, x):
        x = self.first_conv(x) # 40
        x = self.block0(x) # 24
        c1 = self.block1(x) # 32
        x = self.block2(c1) # 32
        x = self.block3(x) # 56
        c2 = self.block4(x)
        x = self.block5(c2)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        c3 = self.block10(x)
        x = self.block11(c3)
        x = self.block12(x)
        c4 = self.block13(x)

        return c1, c2, c3, c4

    @staticmethod
    def build_from_config(config):
        first_conv = set_layer_from_config(config['first_conv'])
        blocks = []
        for index, block_config in enumerate(config['blocks']):
            blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))
        net = ProxylessNASNets(first_conv, blocks)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)
        
        return net
    
    def set_bn_param(self, momentum, eps):
        for m in self.cells():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.momentum = momentum
                m.eps = eps
        return

def ofa_v100_gpu64_6ms():
    net_config = json.load(open(os.path.join(os.path.dirname(__file__), 'net.config'), 'r'))
    return ProxylessNASNets.build_from_config(net_config)