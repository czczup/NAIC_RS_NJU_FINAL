import torch
import torch.nn as nn


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


class IBN_ResNext(nn.Module):
    
    def __init__(self, block, layers, output_stride=8, zero_init_residual=False, groups=1,
                 width_per_group=64, norm_layer=nn.BatchNorm2d, ibn_cfg=('a', 'a', 'a', None)):
        super(IBN_ResNext, self).__init__()
        
        self.c1_channels = 256
        self.c4_channels = 2048
        
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
    
def ibn_b_resnext50_32x4d(norm_layer=nn.BatchNorm2d):
    return IBN_ResNext(IBN_Bottleneck, [3, 4, 6, 3], groups=32,
                   width_per_group=4, norm_layer=norm_layer, ibn_cfg=('b', 'b', 'b', None))
