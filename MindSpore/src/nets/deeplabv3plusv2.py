import mindspore.nn as nn
from mindspore.ops import operations as P
from .resnet import Resnet, Bottleneck
from .module import ASPP, SeparableConv2d, ConvBNReLU, FCNHead
from .ofav100 import ProxylessNASNets, ofa_v100_gpu64_6ms
import mindspore.ops as ops


class DeepLabV3PlusV2(nn.Cell):
    def __init__(self, phase='train', num_classes=[8, 14], output_stride=8, aux=False, freeze_bn=False):
        super(DeepLabV3PlusV2, self).__init__()
        
        self.aux = aux
        self.training = (phase == 'train')
        # self.encoder = Resnet(Bottleneck, [3, 4, 23, 3], output_stride=output_stride)
        use_batch_statistics = not freeze_bn
        self.encoder = ofa_v100_gpu64_6ms(use_batch_statistics)
        if isinstance(self.encoder, Resnet):
            c1_channels, c3_channels, c4_channels = 256, 1024, 2048
        elif isinstance(self.encoder, ProxylessNASNets):
            c1_channels, c3_channels, c4_channels = 32, 128, 248
            
        self.head = DeepLabHead(num_classes=8, c1_channels=c1_channels,
                                c4_channels=c4_channels, use_batch_statistics=use_batch_statistics)
        self.head2 = DeepLabHead(num_classes=14, c1_channels=c1_channels,
                                 c4_channels=c4_channels, use_batch_statistics=use_batch_statistics)
        
        if self.aux and self.training:
            self.auxlayer = FCNHead(c3_channels, num_classes[0], use_batch_statistics=use_batch_statistics)
            self.auxlayer2 = FCNHead(c3_channels, num_classes[1], use_batch_statistics=use_batch_statistics)
        
        self.shape = P.Shape()
        self.add = P.TensorAdd()
        self.transpose = P.Transpose()
        self.softmax = nn.Softmax()
        self.mul = P.Mul()
        self.add1 = ops.TensorAdd()
    
    def train_construct(self, x):
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

    def split_v2(self, x):
        x = P.Split(1, 8)(x)
        x = (x[0], x[1], x[2], x[7], x[7],
             x[3], x[3], x[4], x[4],
             x[5], x[5], x[6], x[6], x[7])
        x = P.Concat(1)(x)
        return x

    def _softmax(self, x):
        x = self.transpose(x,(0,2,3,1))
        x = self.softmax(x)
        x = self.transpose(x,(0,3,1,2))
        return x
    
    def test_construct(self, x):
        size = self.shape(x)
        c1, _, c3, c4 = self.encoder(x)
        
        x1 = self.head(c4, c1)
        x1 = P.ResizeNearestNeighbor((size[2], size[3]), True)(x1)
        x1 = self._softmax(x1)
        x1 = self.split_v2(x1)

        x2 = self.head2(c4, c1)
        x2 = P.ResizeNearestNeighbor((size[2], size[3]), True)(x2)
        x2 = self._softmax(x2)
        x = self.add(x1, x2)
        return x
        
    def construct(self, x):
        if self.training:
            return self.train_construct(x)
        else:
            return self.test_construct(x)
        
        
class DeepLabHead(nn.Cell):
    def __init__(self, num_classes, c1_channels=256, c4_channels=2048, output_stride=8,
                 use_batch_statistics=True):
        super(DeepLabHead, self).__init__()
        self.aspp = ASPP(c4_channels, 256, output_stride, use_batch_statistics=use_batch_statistics)
        last_channels = 256
        self.c1_block = ConvBNReLU(c1_channels, 48, 1, use_batch_statistics=use_batch_statistics)
        last_channels += 48
        self.block = nn.SequentialCell([
            SeparableConv2d(last_channels, 256, 3, use_batch_statistics=use_batch_statistics),
            SeparableConv2d(256, 256, 3, use_batch_statistics=use_batch_statistics),
            nn.Conv2d(256, num_classes, 1, has_bias=True)
        ])
        self.shape = P.Shape()
        self.concat = P.Concat(axis=1)

    def construct(self, x, c1):
        size = self.shape(c1)
        x = self.aspp(x)
        x = P.ResizeNearestNeighbor((size[2], size[3]), True)(x)
        c1 = self.c1_block(c1)

        return self.block(self.concat((x, c1)))
