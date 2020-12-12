from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
try:
    import apex
except:
    print("apex is not installed successfully")
import os


"""
0. Tools
"""
def fuse_conv_bn(conv, bn):
    conv_w = conv.weight
    conv_b = conv.bias if conv.bias is not None else torch.zeros_like(
        bn.running_mean)

    factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    conv.weight = nn.Parameter(conv_w *
                               factor.reshape([conv.out_channels, 1, 1, 1]))
    conv.bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)
    return conv


def fuse_module(m):
    last_conv = None
    last_conv_name = None

    for name, child in m.named_children():
        if isinstance(child, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            fused_conv = fuse_conv_bn(last_conv, child)
            m._modules[last_conv_name] = fused_conv
            m._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, nn.Conv2d):
            last_conv = child
            last_conv_name = name
        else:
            fuse_module(child)
    return m

"""
1. Definition of OFA-V100
"""
class MyModule(nn.Module):
    def forward(self, x):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError


class MyNetwork(MyModule):
    def forward(self, x):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    def set_bn_param(self, momentum, eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.momentum = momentum
                m.eps = eps
        return

def get_same_pad(ksize, dilation):
    pad = ((ksize - 1) * dilation) / 2
    return int(pad)

class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.
    
def build_activation(act_func, inplace=True):
    if act_func == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_func == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'sigmoid':
        return nn.Sigmoid()
    elif act_func == 'h_swish':
        return Hswish(inplace=inplace)
    elif act_func == 'h_sigmoid':
        return Hsigmoid(inplace=inplace)
    elif act_func is None:
        return None
    else:
        raise ValueError('do not support: %s' % act_func)
    
def set_layer_from_config(layer_config):
    if layer_config is None:
        return None

    name2layer = {
        ConvLayer.__name__: ConvLayer,
        IdentityLayer.__name__: IdentityLayer,
        MBInvertedConvLayer.__name__: MBInvertedConvLayer,
    }

    layer_name = layer_config.pop('name')
    layer = name2layer[layer_name]
    return layer.build_from_config(layer_config)


class My2DLayer(MyModule):

    def __init__(self, in_channels, out_channels,
                 use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act'):
        super(My2DLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        """ modules """
        modules = {}
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                modules['bn'] = nn.BatchNorm2d(in_channels)
            else:
                modules['bn'] = nn.BatchNorm2d(out_channels)
        else:
            modules['bn'] = None
        # activation
        modules['act'] = build_activation(self.act_func, self.ops_list[0] != 'act')
        # dropout
        if self.dropout_rate > 0:
            modules['dropout'] = nn.Dropout2d(self.dropout_rate, inplace=True)
        else:
            modules['dropout'] = None
        # weight
        modules['weight'] = self.weight_op()

        # add modules
        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == 'weight':
                # dropout before weight operation
                if modules['dropout'] is not None:
                    self.add_module('dropout', modules['dropout'])
                for key in modules['weight']:
                    self.add_module(key, modules['weight'][key])
            else:
                self.add_module(op, modules[op])

    @property
    def ops_list(self):
        return self.ops_order.split('_')

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self.ops_order)

    def weight_op(self):
        raise NotImplementedError

    """ Methods defined in MyModule """
    def forward(self, x):
        # similar to nn.Sequential
        for module in self._modules.values():
            x = module(x)
        return x

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError


class ConvLayer(My2DLayer):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, dilation=1, groups=1, bias=False, has_shuffle=False,
                 use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act'):
        # default normal 3x3_Conv with bn and relu
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle

        super(ConvLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

    def weight_op(self):
        padding = get_same_pad(self.kernel_size, self.dilation)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        weight_dict = OrderedDict()
        weight_dict['conv'] = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=padding,
            dilation=self.dilation, groups=self.groups, bias=self.bias
        )
        return weight_dict

    @staticmethod
    def build_from_config(config):
        return ConvLayer(**config)


class IdentityLayer(My2DLayer):

    def __init__(self, in_channels, out_channels,
                 use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act'):
        super(IdentityLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

    def weight_op(self):
        return None

    @staticmethod
    def build_from_config(config):
        return IdentityLayer(**config)
    

class MBInvertedConvLayer(MyModule):

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
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(self.in_channels, feature_dim, 1, 1, 0, bias=False)),
                ('bn', nn.BatchNorm2d(feature_dim)),
                ('act', build_activation(self.act_func, inplace=True)),
            ]))

        pad = get_same_pad(self.kernel_size, dilation=self.dilation)
        depth_conv_modules = [
            ('conv', nn.Conv2d(feature_dim, feature_dim, kernel_size, stride, pad,
                               groups=feature_dim, bias=False, dilation=dilation)),
            ('bn', nn.BatchNorm2d(feature_dim)),
            ('act', build_activation(self.act_func, inplace=True))
        ]
        self.depth_conv = nn.Sequential(OrderedDict(depth_conv_modules))

        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
        ]))

    def forward(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    @staticmethod
    def build_from_config(config):
        return MBInvertedConvLayer(**config)


class MobileInvertedResidualBlock(MyModule):
    
    def __init__(self, mobile_inverted_conv, shortcut, ibn=False):
        super(MobileInvertedResidualBlock, self).__init__()
        
        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut
        if ibn and self.mobile_inverted_conv.config['out_channels'] <= 32:
            self.IN = nn.InstanceNorm2d(self.mobile_inverted_conv.config['out_channels'])
        else:
            self.IN = None
    
    def forward(self, x):
        if self.mobile_inverted_conv is None:
            res = x
        elif self.shortcut is None:
            res = self.mobile_inverted_conv(x)
        else:
            res = self.mobile_inverted_conv(x) + self.shortcut(x)
        if self.IN is not None:
            res = self.IN(res)
        return res
    
    @staticmethod
    def build_from_config(config, ibn=False):
        mobile_inverted_conv = set_layer_from_config(config['mobile_inverted_conv'])
        shortcut = set_layer_from_config(config['shortcut'])
        return MobileInvertedResidualBlock(mobile_inverted_conv, shortcut, ibn=ibn)


class OFA_V100(MyNetwork):
    
    def __init__(self, first_conv, blocks):
        super(OFA_1080Ti_Tiny, self).__init__()
        
        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, x):
        outs = list()
        x = self.first_conv(x)
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index == 2 or index == 4 or index == 10:
                outs.append(x)
        outs.append(x)
        return outs
    
    @staticmethod
    def build_from_config(config, ibn=False):
        first_conv = set_layer_from_config(config['first_conv'])
        
        blocks = []
        for index, block_config in enumerate(config['blocks']):
            blocks.append(MobileInvertedResidualBlock.build_from_config(block_config, ibn=ibn))
        
        net = OFA_1080Ti_Tiny(first_conv, blocks)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)
        
        return net


class OFA_1080Ti_Tiny(MyNetwork):
    
    def __init__(self, first_conv, blocks):
        super(OFA_1080Ti_Tiny, self).__init__()
        
        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, x):
        outs = list()
        x = self.first_conv(x)
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index == 2 or index == 6 or index == 12:
                outs.append(x)
        outs.append(x)
        return outs
    
    @staticmethod
    def build_from_config(config, ibn=False):
        first_conv = set_layer_from_config(config['first_conv'])
        
        blocks = []
        for index, block_config in enumerate(config['blocks']):
            blocks.append(MobileInvertedResidualBlock.build_from_config(block_config, ibn=ibn))

        net = OFA_1080Ti_Tiny(first_conv, blocks)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)
        
        return net


def ofa_v100_gpu64_6ms(norm_layer=nn.BatchNorm2d):
    net_config = json.load(open(os.path.join(os.path.dirname(__file__), 'ofav100.config'), 'r'))
    return OFA_V100.build_from_config(net_config)

def ofa_1080ti_gpu64_27ms_tiny(norm_layer=nn.BatchNorm2d):
    net_config = json.load(open(os.path.join(os.path.dirname(__file__), 'ofa1080ti.config'), 'r'))
    return OFA_1080Ti_Tiny.build_from_config(net_config)


"""
2. Definition of DeepLabV3Plus
"""
class DeepLabV3Plus(nn.Module):
    def __init__(self, nclass=[8, 14], get_backbone=None, channels=None):
        super(DeepLabV3Plus, self).__init__()
        self.nclass = nclass
        self.encoder = get_backbone()
        self.norm_layer = nn.BatchNorm2d
        self.head = _DeepLabHead(self.nclass[0], c1_channels=channels[0], c4_channels=channels[1])
        self.head2 = _DeepLabHead(self.nclass[1], c1_channels=channels[0], c4_channels=channels[1])
        
        self.__setattr__('decoder', ['head', 'head2'])
    
    def split(self, x):
        x = torch.split(x, 1, dim=1)
        x = [x[0], x[1], x[2], x[7], x[7],
             x[3], x[3], x[4], x[4],
             x[5], x[5], x[6], x[6], x[7]]
        input = torch.cat(x, dim=1)
        return input
    
    def forward_8_14_to_14_v2(self, x):
        c1, _, c3, c4 = self.encoder(x)
        outputs = list()
        x1 = self.head(c4, c1)  # 8
        x1 = F.softmax(x1, dim=1)
        x1 = self.split(x1)  # 14
        x2 = self.head2(c4, c1)  # 14
        x2 = F.softmax(x2, dim=1)
        x = x1 + x2  # 14
        outputs.append(x)
        return tuple(outputs)
    
    def forward(self, x):
        return self.forward_8_14_to_14_v2(x)


class _DeepLabHead(nn.Module):
    def __init__(self, nclass, c1_channels=256, c4_channels=2048, norm_layer=nn.BatchNorm2d):
        super(_DeepLabHead, self).__init__()
        self.aspp = _ASPP(c4_channels, 256)
        last_channels = 256
        self.c1_block = _ConvBNReLU(c1_channels, 48, 1, norm_layer=norm_layer)
        last_channels += 48
        self.block = nn.Sequential(
            SeparableConv2d(last_channels, 256, 3, norm_layer=norm_layer, relu_first=False),
            SeparableConv2d(256, 256, 3, norm_layer=norm_layer, relu_first=False),
            nn.Conv2d(256, nclass, 1))
    
    def forward(self, x, c1):
        size = c1.size()[2:]
        x = self.aspp(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        c1 = self.c1_block(c1)
        return self.block(torch.cat([x, c1], dim=1))


class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, relu_first=True,
                 bias=False, norm_layer=nn.BatchNorm2d):
        super().__init__()
        depthwise = nn.Conv2d(inplanes, inplanes, kernel_size,
                              stride=stride, padding=dilation,
                              dilation=dilation, groups=inplanes, bias=bias)
        bn_depth = norm_layer(inplanes)
        pointwise = nn.Conv2d(inplanes, planes, 1, bias=bias)
        bn_point = norm_layer(planes)
        
        if relu_first:
            self.block = nn.Sequential(OrderedDict([('relu', nn.ReLU()),
                                                    ('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point)
                                                    ]))
        else:
            self.block = nn.Sequential(OrderedDict([('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('relu1', nn.ReLU(inplace=True)),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point),
                                                    ('relu2', nn.ReLU(inplace=True))
                                                    ]))
    
    def forward(self, x):
        return self.block(x)


class _ASPP(nn.Module):
    def __init__(self, in_channels=2048, out_channels=256, output_stride=8):
        super().__init__()
        if output_stride == 16:
            dilations = [6, 12, 18]
        elif output_stride == 8:
            dilations = [12, 24, 36]
        elif output_stride == 32:
            dilations = [6, 12, 18]
        else:
            raise NotImplementedError
        
        self.aspp0 = nn.Sequential(OrderedDict([('conv', nn.Conv2d(in_channels, out_channels, 1, bias=False)),
                                                ('bn', nn.BatchNorm2d(out_channels)),
                                                ('relu', nn.ReLU(inplace=True))]))
        self.aspp1 = SeparableConv2d(in_channels, out_channels, dilation=dilations[0], relu_first=False)
        self.aspp2 = SeparableConv2d(in_channels, out_channels, dilation=dilations[1], relu_first=False)
        self.aspp3 = SeparableConv2d(in_channels, out_channels, dilation=dilations[2], relu_first=False)
        
        self.image_pooling = nn.Sequential(OrderedDict([('gap', nn.AdaptiveAvgPool2d((1, 1))),
                                                        ('conv', nn.Conv2d(in_channels, out_channels, 1, bias=False)),
                                                        ('bn', nn.BatchNorm2d(out_channels)),
                                                        ('relu', nn.ReLU(inplace=True))]))
        
        self.conv = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.1)
    
    def forward(self, x):
        pool = self.image_pooling(x)
        pool = F.interpolate(pool, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x = torch.cat((pool, x0, x1, x2, x3), dim=1)
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


def init_model():
    device = torch.device("cuda")
    # model = DeepLabV3Plus(nclass=[8, 14], get_backbone=ofa_v100_gpu64_6ms, channels=[32, 248])
    model = DeepLabV3Plus(nclass=[8, 14], get_backbone=ofa_1080ti_gpu64_27ms_tiny, channels=[32, 416])

    model_path = os.path.join(os.path.dirname(__file__), "best_model.pth")
    print('load test model from {}'.format(model_path))

    model = fuse_module(model)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict_ = dict()
    for k, v in state_dict.items():
        if v.dtype == torch.qint8:
            state_dict_[k] = v.dequantize()
        else:
            state_dict_[k] = v
    msg = model.load_state_dict(state_dict_, strict=False)
    
    print(msg)
    model.eval()
    model = model.to(device)
    
    try:
        model = apex.amp.initialize(model.cuda(), opt_level="O3")
    except:
        pass
    
    return model