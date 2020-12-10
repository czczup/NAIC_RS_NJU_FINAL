import mindspore.nn as nn
from mindspore.ops import operations as P
from .resnet import Resnet, Bottleneck
from .module import ASPP, SeparableConv2d, ConvBNReLU, FCNHead


class DeepLabV3PlusV2(nn.Cell):
    def __init__(self, phase='train', num_classes=[8, 14], output_stride=8, freeze_bn=False, aux=False):
        super(DeepLabV3PlusV2, self).__init__()

        c1_channels = 256
        c3_channels = 1024
        c4_channels = 2048
        
        self.aux = aux
        self.training = (phase == 'train')
        self.encoder = Resnet(Bottleneck, [3, 4, 23, 3], output_stride=output_stride)
        self.head = DeepLabHead(num_classes[0], c1_channels=c1_channels, c4_channels=c4_channels)
        self.head2 = DeepLabHead(num_classes[1], c1_channels=c1_channels, c4_channels=c4_channels)
        
        if self.aux and self.training:
            self.auxlayer = FCNHead(c3_channels, num_classes[0])
            self.auxlayer2 = FCNHead(c3_channels, num_classes[1])
        
        self.shape = P.Shape()

    def construct(self, x):
        size = self.shape(x)
        c1, _, c3, c4 = self.encoder(x)

        outputs1 = list()
        x = self.head(c4, c1)
        x = P.ResizeNearestNeighbor((size[2], size[3]), True)(x)
        outputs1.append(x)

        if self.aux and self.training:
            auxout = self.auxlayer(c3)
            auxout = P.ResizeNearestNeighbor((size[2], size[3]), True)(auxout)
            outputs1.append(auxout)

        outputs2 = list()
        x = self.head2(c4, c1)
        x = P.ResizeNearestNeighbor((size[2], size[3]), True)(x)
        outputs2.append(x)
        if self.aux and self.training:
            auxout = self.auxlayer2(c3)
            auxout = P.ResizeNearestNeighbor((size[2], size[3]), True)(auxout)
            outputs2.append(auxout)
        return tuple([outputs1, outputs2])


class DeepLabHead(nn.Cell):
    def __init__(self, num_classes, c1_channels=256, c4_channels=2048, output_stride=8):
        super(DeepLabHead, self).__init__()
        self.aspp = ASPP(c4_channels, 256, output_stride)
        last_channels = 256
        self.c1_block = ConvBNReLU(c1_channels, 48, 1)
        last_channels += 48
        self.block = nn.SequentialCell([
            SeparableConv2d(last_channels, 256, 3),
            SeparableConv2d(256, 256, 3),
            nn.Conv2d(256, num_classes, 1)
        ])
        self.shape = P.Shape()
        self.concat = P.Concat(axis=1)

    def construct(self, x, c1):
        size = self.shape(c1)
        x = self.aspp(x)
        x = P.ResizeNearestNeighbor((size[2], size[3]), True)(x)
        c1 = self.c1_block(c1)
        
        return self.block(self.concat((x, c1)))

