from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import apex
except:
    print("apex is not installed successfully")
import time

"""
1. Definition of ResNeXt (ResNeXt50_32x4d and ResNeXt101_32x8d)
"""
class IBN(nn.Module):
    def __init__(self, planes, ratio=0.5):
        super(IBN, self).__init__()
        self.half = int(planes * (1-ratio))
        self.BN = nn.BatchNorm2d(self.half)
        self.IN = nn.InstanceNorm2d(planes - self.half, affine=True)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.BN(split[0].contiguous())
        out2 = self.IN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class IBN_Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, ibn=None, **kwargs):
        super(IBN_Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        
        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        
        if ibn == 'a':
            self.bn1 = IBN(width)
        else:
            self.bn1 = norm_layer(width)
        
        self.conv2 = nn.Conv2d(width, width, 3, stride, dilation, dilation, groups, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.IN = nn.InstanceNorm2d(planes * 4, affine=True) if ibn == 'b' else None
        self.relu = nn.ReLU(True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        if self.IN is not None:
            out = self.IN(out)
        out = self.relu(out)
        
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, **kwargs):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride, dilation, dilation, groups, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class IBN_ResNext(nn.Module):

    def __init__(self, block, layers, output_stride=8, zero_init_residual=False, groups=1,
                 width_per_group=64, norm_layer=nn.BatchNorm2d, ibn_cfg=('a', 'a', 'a', None)):
        super(IBN_ResNext, self).__init__()
        if output_stride == 32:
            dilations = [1, 1]
            strides = [2, 2]
        elif output_stride == 16:
            dilations = [1, 2]
            strides = [2, 1]
        elif output_stride == 8:
            dilations = [2, 4]
            strides = [1, 1]
        else:
            raise NotImplementedError

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = nn.Conv2d(3, self.inplanes, 7, 2, 3, bias=False)

        if ibn_cfg[0] == 'b':
            self.bn1 = nn.InstanceNorm2d(64, affine=True)
        else:
            self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, ibn=ibn_cfg[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer, ibn=ibn_cfg[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[0], dilation=dilations[0],
                                       norm_layer=norm_layer, ibn=ibn_cfg[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[1], dilation=dilations[1],
                                       norm_layer=norm_layer, ibn=ibn_cfg[3])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=nn.BatchNorm2d, ibn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                norm_layer(planes * block.expansion)
            )

        layers = list()
        if dilation in (1, 2):
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, norm_layer=norm_layer, ibn=ibn))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, dilation=2, norm_layer=norm_layer, ibn=ibn))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                dilation=dilation, norm_layer=norm_layer, ibn=ibn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        return c1, c2, c3, c4

class ResNext(nn.Module):
    def __init__(self, block, layers, output_stride=8, zero_init_residual=False, groups=1,
                 width_per_group=64, norm_layer=nn.BatchNorm2d):
        super(ResNext, self).__init__()
        if output_stride == 32:
            dilations = [1, 1]
            strides = [2, 2]
        elif output_stride == 16:
            dilations = [1, 2]
            strides = [2, 1]
        elif output_stride == 8:
            dilations = [2, 4]
            strides = [1, 1]
        else:
            raise NotImplementedError

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, 7, 2, 3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[0], dilation=dilations[0], norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[1], dilation=dilations[1], norm_layer=norm_layer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                norm_layer(planes * block.expansion)
            )

        layers = list()
        if dilation in (1, 2):
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, dilation=2, norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                dilation=dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c1, c2, c3, c4

def resnext50_32x4d():
    return ResNext(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4, output_stride=8)

def ibn_b_resnext50_32x4d(norm_layer=nn.BatchNorm2d):
    return IBN_ResNext(IBN_Bottleneck, [3, 4, 6, 3], groups=32,
                   width_per_group=4, norm_layer=norm_layer, ibn_cfg=('b', 'b', 'b', None))


class IBN_BottleneckV1b(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=nn.BatchNorm2d, ibn=None):
        super(IBN_BottleneckV1b, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride,
                               dilation, dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.IN = nn.InstanceNorm2d(planes * 4, affine=True) if ibn == 'b' else None
        self.relu = nn.ReLU(True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        if self.IN is not None:
            out = self.IN(out)
        out = self.relu(out)
        
        return out


class IBN_ResNet(nn.Module):
    
    def __init__(self, block, layers, num_classes=1000, deep_stem=False,
                 norm_layer=nn.BatchNorm2d, ibn_cfg=('b', 'b', 'b', None)):
        output_stride = 8
        scale = 1
        if output_stride == 32:
            dilations = [1, 1]
            strides = [2, 2]
        elif output_stride == 16:
            dilations = [1, 2]
            strides = [2, 1]
        elif output_stride == 8:
            dilations = [2, 4]
            strides = [1, 1]
        else:
            raise NotImplementedError
        self.inplanes = int((128 if deep_stem else 64) * scale)
        super(IBN_ResNet, self).__init__()
        if deep_stem:
            # resnet vc
            mid_channel = int(64 * scale)
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, mid_channel, 3, 2, 1, bias=False),
                norm_layer(mid_channel),
                nn.ReLU(True),
                nn.Conv2d(mid_channel, mid_channel, 3, 1, 1, bias=False),
                norm_layer(mid_channel),
                nn.ReLU(True),
                nn.Conv2d(mid_channel, self.inplanes, 3, 1, 1, bias=False)
            )
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, 7, 2, 3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(block, int(64 * scale), layers[0], norm_layer=norm_layer, ibn=ibn_cfg[0])
        self.layer2 = self._make_layer(block, int(128 * scale), layers[1], stride=2, norm_layer=norm_layer,
                                       ibn=ibn_cfg[1])
        
        self.layer3 = self._make_layer(block, int(256 * scale), layers[2], stride=strides[0], dilation=dilations[0],
                                       norm_layer=norm_layer, ibn=ibn_cfg[2])
        self.layer4 = self._make_layer(block, int(512 * scale), layers[3], stride=strides[1], dilation=dilations[1],
                                       norm_layer=norm_layer, ibn=ibn_cfg[3])
        
        self.last_inp_channels = int(512 * block.expansion * scale)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(512 * block.expansion * scale), num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=nn.BatchNorm2d, ibn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                norm_layer(planes * block.expansion),
            )
        
        layers = []
        if dilation in (1, 2):
            layers.append(block(self.inplanes, planes, stride, dilation=1, downsample=downsample,
                                previous_dilation=dilation, norm_layer=norm_layer, ibn=ibn))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2, downsample=downsample,
                                previous_dilation=dilation, norm_layer=norm_layer, ibn=ibn))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))
        
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                                previous_dilation=dilation, norm_layer=norm_layer, ibn=ibn))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        
        return c1, c2, c3, c4

def ibn_resnet50(norm_layer=nn.BatchNorm2d):
    num_block = [3, 4, 6, 3]
    return IBN_ResNet(IBN_BottleneckV1b, num_block, norm_layer=norm_layer, ibn_cfg=('b', 'b', 'b', None))

"""
2. Definition of DeepLabV3Plus
"""
class DeepLabV3Plus(nn.Module):
    def __init__(self, nclass=[8, 14], get_backbone=None):
        super(DeepLabV3Plus, self).__init__()
        self.nclass = nclass
        self.encoder = get_backbone()
        self.norm_layer = nn.BatchNorm2d
        self.head = _DeepLabHead(self.nclass[0], c1_channels=256, c4_channels=2048)
        self.head2 = _DeepLabHead(self.nclass[1], c1_channels=256, c4_channels=2048)

        self.__setattr__('decoder', ['head', 'head2'])

    def merge(self, x):
        x = torch.split(x, 1, dim=1)
        x = [x[0], x[1], x[2], torch.add(x[5], x[6]),
                 torch.add(x[7], x[8]), torch.add(x[9], x[10]),
                 torch.add(x[11], x[12]),
                 torch.add(torch.add(x[3], x[4]), x[13])]
        input = torch.cat(x, dim=1)
        return input
    
    def split(self, x):
        x = torch.split(x, 1, dim=1)
        x = [x[0], x[1], x[2], x[7]/3.0, x[7]/3.0,
             x[3]/2.0, x[3]/2.0, x[4]/2.0, x[4]/2.0,
             x[5]/2.0, x[5]/2.0, x[6]/2.0, x[6]/2.0,
             x[7]/3.0]
        input = torch.cat(x, dim=1)
        return input
    
    def split_v2(self, x):
        x = torch.split(x, 1, dim=1)
        x = [x[0], x[1], x[2], x[7], x[7],
             x[3], x[3], x[4], x[4],
             x[5], x[5], x[6], x[6], x[7]]
        input = torch.cat(x, dim=1)
        return input
    
    def forward_8_to_8(self, x):
        c1, _, c3, c4 = self.encoder(x)
        outputs = list()
        x = self.head(c4, c1)
        outputs.append(x)
        return tuple(outputs)
    
    def forward_14_to_14(self, x):
        c1, _, c3, c4 = self.encoder(x)
        outputs = list()
        x = self.head2(c4, c1)
        outputs.append(x)
        return tuple(outputs)
    
    def forward_14_to_8(self, x):
        c1, _, c3, c4 = self.encoder(x)
        outputs = list()
        x = self.head2(c4, c1)
        x = F.softmax(x, dim=1)
        x = self.merge(x)
        outputs.append(x)
        return tuple(outputs)
    
    def forward_8_14_to_8(self, x):
        c1, _, c3, c4 = self.encoder(x)
        outputs = list()
        x1 = self.head(c4, c1)
        x1 = F.softmax(x1, dim=1)
        x2 = self.head2(c4, c1)
        x2 = F.softmax(x2, dim=1)
        x2 = self.merge(x2)
        x = x1 + x2
        outputs.append(x)
        return tuple(outputs)
    
    def forward_8_14_to_14(self, x):
        c1, _, c3, c4 = self.encoder(x)
        outputs = list()
        x1 = self.head(c4, c1)
        x1 = F.softmax(x1, dim=1)
        x1 = self.split(x1)
        x2 = self.head2(c4, c1)
        x2 = F.softmax(x2, dim=1)
        x = x1 + x2
        outputs.append(x)
        return tuple(outputs)
    
    def forward_8_14_to_14_v2(self, x):
        c1, _, c3, c4 = self.encoder(x)
        outputs = list()
        x1 = self.head(c4, c1)
        x1 = F.softmax(x1, dim=1)
        x1 = self.split_v2(x1)
        x2 = self.head2(c4, c1)
        x2 = F.softmax(x2, dim=1)
        x = x1 + x2
        outputs.append(x)
        return tuple(outputs)

    def forward(self, x, mode):
        if mode == "01":
            return self.forward_8_to_8(x)
        elif mode == "02":
            return self.forward_14_to_8(x)
        elif mode == "03":
            return self.forward_8_14_to_8(x)
        elif mode == "04":
            return self.forward_14_to_14(x)
        elif mode == "05":
            return self.forward_8_14_to_14(x)
        elif mode == "06":
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

        self.conv = nn.Conv2d(out_channels*5, out_channels, 1, bias=False)
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


"""
4. Definition of DUNet
"""
class DUNet(nn.Module):
    """Decoders Matter for Semantic Segmentation
    Reference:
        Zhi Tian, Tong He, Chunhua Shen, and Youliang Yan.
        "Decoders Matter for Semantic Segmentation:
        Data-Dependent Decoding Enables Flexible Feature Aggregation." CVPR, 2019
    """
    def __init__(self, nclass=[8, 14], get_backbone=None):
        super(DUNet, self).__init__()
        self.nclass = nclass
        self.encoder = get_backbone()
        self.norm_layer = nn.BatchNorm2d
        self.head = _DUHead(2144, norm_layer=self.norm_layer)
        self.head2 = _DUHead(2144, norm_layer=self.norm_layer)
        self.dupsample = DUpsampling(256, 8, scale_factor=2)
        self.dupsample2 = DUpsampling(256, 14, scale_factor=2)
        
        self.__setattr__('decoder', ['dupsample', 'dupsample2', 'head', 'head2'])
    
    def merge(self, x):
        x = torch.split(x, 1, dim=1)
        x = [x[0], x[1], x[2], torch.add(x[5], x[6]),
             torch.add(x[7], x[8]), torch.add(x[9], x[10]),
             torch.add(x[11], x[12]),
             torch.add(torch.add(x[3], x[4]), x[13])]
        input = torch.cat(x, dim=1)
        return input
    
    def split(self, x):
        x = torch.split(x, 1, dim=1)
        x = [x[0], x[1], x[2], x[7] / 3.0, x[7] / 3.0,
             x[3] / 2.0, x[3] / 2.0, x[4] / 2.0, x[4] / 2.0,
             x[5] / 2.0, x[5] / 2.0, x[6] / 2.0, x[6] / 2.0,
             x[7] / 3.0]
        input = torch.cat(x, dim=1)
        return input
    
    def split_v2(self, x):
        x = torch.split(x, 1, dim=1)
        x = [x[0], x[1], x[2], x[7], x[7],
             x[3], x[3], x[4], x[4],
             x[5], x[5], x[6], x[6], x[7]]
        input = torch.cat(x, dim=1)
        return input
    
    def forward_8_to_8(self, x):
        _, c2, c3, c4 = self.encoder(x)
        outputs = list()
        x = self.head(c2, c3, c4)  # 256
        x = self.dupsample(x)  # 8
        x = F.softmax(x, dim=1)
        outputs.append(x)
        return tuple(outputs)
    
    def forward_14_to_14(self, x):
        _, c2, c3, c4 = self.encoder(x)
        outputs = list()
        x = self.head2(c2, c3, c4)  # 256
        x = self.dupsample2(x)  # 14
        x = F.softmax(x, dim=1)
        outputs.append(x)
        return tuple(outputs)
    
    def forward_14_to_8(self, x):
        _, c2, c3, c4 = self.encoder(x)
        outputs = list()
        x = self.head2(c2, c3, c4)  # 256
        x = self.dupsample2(x)  # 14
        x = F.softmax(x, dim=1)
        x = self.merge(x)  # 8
        outputs.append(x)
        return tuple(outputs)
    
    def forward_8_14_to_8(self, x):
        c_, c2, c3, c4 = self.encoder(x)
        outputs = list()
        x1 = self.head(c2, c3, c4)  # 256
        x1 = self.dupsample(x1)  # 8
        x1 = F.softmax(x1, dim=1)  # 8
        
        x2 = self.head2(c2, c3, c4)  # 256
        x2 = self.dupsample2(x2)  # 14
        x2 = F.softmax(x2, dim=1)  # 14
        x2 = self.merge(x2)  # 8
        
        x = x1 + x2
        outputs.append(x)
        return tuple(outputs)
    
    def forward_8_14_to_14(self, x):
        c_, c2, c3, c4 = self.encoder(x)
        outputs = list()
        x1 = self.head(c2, c3, c4)  # 256
        x1 = self.dupsample(x1)  # 8
        x1 = F.softmax(x1, dim=1)  # 8
        x1 = self.split(x1)  # 14
        
        x2 = self.head2(c2, c3, c4)  # 256
        x2 = self.dupsample2(x2)  # 14
        x2 = F.softmax(x2, dim=1)  # 14
        x = x1 + x2
        outputs.append(x)
        return tuple(outputs)
    
    def forward_8_14_to_14_v2(self, x):
        c_, c2, c3, c4 = self.encoder(x)
        outputs = list()
        x1 = self.head(c2, c3, c4)  # 256
        x1 = self.dupsample(x1)  # 8
        x1 = F.softmax(x1, dim=1)  # 8
        x1 = self.split_v2(x1)  # 14
    
        x2 = self.head2(c2, c3, c4)  # 256
        x2 = self.dupsample2(x2)  # 14
        x2 = F.softmax(x2, dim=1)  # 14
        x = x1 + x2
        outputs.append(x)
        return tuple(outputs)

    def forward(self, x, mode):
        if mode == "01":
            return self.forward_8_to_8(x)
        elif mode == "02":
            return self.forward_14_to_8(x)
        elif mode == "03":
            return self.forward_8_14_to_8(x)
        elif mode == "04":
            return self.forward_14_to_14(x)
        elif mode == "05":
            return self.forward_8_14_to_14(x)
        elif mode == "06":
            return self.forward_8_14_to_14_v2(x)


class FeatureFused(nn.Module):
    """Module for fused features"""
    
    def __init__(self, inter_channels=48, norm_layer=nn.BatchNorm2d):
        super(FeatureFused, self).__init__()
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, inter_channels, 1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1024, inter_channels, 1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True)
        )
    
    def forward(self, c2, c3, c4):
        size = c4.size()[2:]
        c2 = self.conv2(F.interpolate(c2, size, mode='bilinear', align_corners=True))
        c3 = self.conv3(F.interpolate(c3, size, mode='bilinear', align_corners=True))
        fused_feature = torch.cat([c4, c3, c2], dim=1)
        return fused_feature


class _DUHead(nn.Module):
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d):
        super(_DUHead, self).__init__()
        self.fuse = FeatureFused(norm_layer=norm_layer)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            norm_layer(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            norm_layer(256),
            nn.ReLU(True)
        )
    
    def forward(self, c2, c3, c4):
        fused_feature = self.fuse(c2, c3, c4)
        out = self.block(fused_feature)
        return out


class DUpsampling(nn.Module):
    """DUsampling module"""
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(DUpsampling, self).__init__()
        self.scale_factor = scale_factor
        self.conv_w = nn.Conv2d(in_channels, out_channels * scale_factor * scale_factor, 1, bias=False)
    
    def forward(self, x):
        x = self.conv_w(x)
        n, c, h, w = x.size()
        # N, C, H, W --> N, W, H, C
        x = x.permute(0, 3, 2, 1).contiguous()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, h * self.scale_factor, c // self.scale_factor)
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, h * self.scale_factor, w * self.scale_factor, c // (self.scale_factor * self.scale_factor))
        # N, H * scale, W * scale, C // (scale ** 2) -- > N, C // (scale ** 2), H * scale, W * scale
        x = x.permute(0, 3, 1, 2)
        return x


"""
5. Init Model
"""
def init_model(model_filename):
    device = torch.device("cuda")
    model = DeepLabV3Plus(nclass=[8,14], get_backbone=ibn_resnet50)
    # model = DUNet(nclass=[8,14], get_backbone=ibn_b_resnext50_32x4d)

    model_path = "../model/%s" % model_filename
    print('load test model from {}'.format(model_path))
    msg = model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage), strict=False)
    print(msg)
    model.eval()
    model = model.to(device)

    try:
        model = apex.amp.initialize(model.cuda(), opt_level="O1")
    except:
        pass
    
    return model
