from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLabV3Plus(nn.Module):
    def __init__(self, nclass=[8, 14], get_backbone=None, channels=None):
        super(DeepLabV3Plus, self).__init__()
        self.nclass = nclass
        self.encoder = get_backbone()
        self.norm_layer = nn.BatchNorm2d
        
        self.head = _DeepLabHead(self.nclass[0], c1_channels=channels[0], c4_channels=channels[1])
        self.head2 = _DeepLabHead(self.nclass[1], c1_channels=channels[0], c4_channels=channels[1])
        
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