import mindspore.nn as nn
from mindspore.ops import operations as P
from src.nets.backbones.resnet import Resnet
from src.nets.backbones.ofav100 import ProxylessNASNets, ofa_v100_gpu64_6ms
from .module import ConvBNReLU, SeparableConv2d, ASPP, FCNHead


class DeepLabV3Plus(nn.Cell):
    def __init__(self, phase='train', num_classes=14, output_stride=8, aux=False):
        super(DeepLabV3Plus, self).__init__()

        self.aux = aux
        self.training = (phase == 'train')
        # self.encoder = Resnet(Bottleneck, [3, 4, 23, 3], output_stride=output_stride)
        self.encoder = ofa_v100_gpu64_6ms()
        if isinstance(self.encoder, Resnet):
            c1_channels, c3_channels, c4_channels = 256, 1024, 2048
        elif isinstance(self.encoder, ProxylessNASNets):
            c1_channels, c3_channels, c4_channels = 32, 128, 248
        
        self.head = DeepLabHead(num_classes, c1_channels=c1_channels, c4_channels=c4_channels)
        self.shape = P.Shape()
        
        if self.aux and self.training:
            self.auxlayer = FCNHead(c3_channels, num_classes)


    def construct(self, x):
        size = self.shape(x)
        
        c1, _, c3, c4 = self.encoder(x)
        x = self.head(c4, c1)
        x = P.ResizeNearestNeighbor((size[2], size[3]), True)(x)
        if self.aux and self.training:
            auxout = self.auxlayer(c3)
            auxout = P.ResizeNearestNeighbor((size[2], size[3]), True)(auxout)
            return x, auxout
        else:
            return x


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

