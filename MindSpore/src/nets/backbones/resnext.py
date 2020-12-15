import mindspore.nn as nn
from mindspore.ops.operations import TensorAdd, Split, Concat
from mindspore.ops import operations as P



class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, **kwargs):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = nn.Conv2d(inplanes, width, 1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        if dilation == 1:
            self.pad2 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT") # pytorch-pad
            self.conv2 = nn.Conv2d(width, width, 3, stride, pad_mode='valid', dilation=dilation,
                                   group=groups, has_bias=False)
        else:
            self.conv2 = nn.Conv2d(width, width, 3, stride, pad_mode='same', dilation=dilation,
                                   group=groups, has_bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        self.add = TensorAdd()
        
    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        if self.dilation == 1:
            out = self.pad2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class ResNext(nn.Cell):

    def __init__(self, block, layers, groups=1, width_per_group=64):
        super(ResNext, self).__init__()

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        
        self.pad1 = nn.Pad(paddings=((0, 0), (0, 0), (3, 2), (3, 2)), mode="CONSTANT") # pytorch-pad
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, pad_mode='valid', has_bias=False) # padding=3
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU()
        self.pad2 = nn.Pad(paddings=((0, 0), (0, 0), (1, 0), (1, 0)), mode="CONSTANT") # pytorch-pad
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, has_bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = list()
        if dilation in (1, 2):
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, dilation=2))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                dilation=dilation))

        return nn.SequentialCell(*layers)

    def construct(self, x):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pad2(x)
        x = self.maxpool(x)
        #
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        return c1, c2, c3, c4


def resnext50_32x4d():
    return ResNext(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4)


def resnext101_32x8d():
    return ResNext(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=8)

