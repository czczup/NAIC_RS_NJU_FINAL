import mindspore.nn as nn
from mindspore.ops import operations as P
from src.nets.module import FCNHead


class DUNetV2(nn.Cell):

    def __init__(self, phase='train', num_classes=[8, 14], aux=False, get_backbone=None, mode="03"):
        super(DUNetV2, self).__init__()
        self.aux = aux
        self.training = (phase == 'train')
        self.nclass = num_classes
        self.encoder = get_backbone()

        self.head = DUHead(2144)
        self.dupsample = DUpsampling(256, 8, scale_factor=2)
        self.head2 = DUHead(2144)
        self.dupsample2 = DUpsampling(256, 14, scale_factor=2)

        if self.aux:
            self.auxlayer = FCNHead(1024, 256)
            self.aux_dupsample = DUpsampling(256, 8, scale_factor=2)
            self.auxlayer2 = FCNHead(1024, 256)
            self.aux_dupsample2 = DUpsampling(256, 14, scale_factor=2)

        self.shape = P.Shape()
        self.add = P.TensorAdd()
        self.transpose = P.Transpose()
        self.softmax = nn.Softmax()
        self.mode = mode


    def train_construct(self, x):
        size = self.shape(x)

        _, c2, c3, c4 = self.encoder(x)

        x1 = self.head(c2, c3, c4)
        x1 = self.dupsample(x1)
        x1 = P.ResizeNearestNeighbor((size[2], size[3]), False)(x1)

        x2 = self.head2(c2, c3, c4)
        x2 = self.dupsample2(x2)
        x2 = P.ResizeNearestNeighbor((size[2], size[3]), False)(x2)
    
        if self.aux and self.training:
            auxout1 = self.auxlayer(c3)
            auxout1 = self.aux_dupsample(auxout1)
            auxout1 = P.ResizeNearestNeighbor((size[2], size[3]), False)(auxout1)

            auxout2 = self.auxlayer2(c3)
            auxout2 = self.aux_dupsample2(auxout2)
            auxout2 = P.ResizeNearestNeighbor((size[2], size[3]), False)(auxout2)
            return x1, auxout1, x2, auxout2
        
        return x1, x2

    def split_v2(self, x):
        x = P.Split(1, 8)(x)
        x = (x[0], x[1], x[2], x[7], x[7],
             x[3], x[3], x[4], x[4],
             x[5], x[5], x[6], x[6], x[7])
        x = P.Concat(1)(x)
        return x

    def _merge(self, x):
        x = P.Split(1, 14)(x)
        x = (x[0], x[1], x[2], self.add(x[5], x[6]),
             self.add(x[7], x[8]), self.add(x[9], x[10]),
             self.add(x[11], x[12]),
             self.add(self.add(x[3], x[4]), x[13]))
        x = P.Concat(1)(x)
        return x

    def _softmax(self, x):
        x = self.transpose(x, (0, 2, 3, 1))
        x = self.softmax(x)
        x = self.transpose(x, (0, 3, 1, 2))
        return x

    def construct_8_14_to_14(self, x):
        size = self.shape(x)
        _, c2, c3, c4 = self.encoder(x)
    
        x1 = self.head(c2, c3, c4)
        x1 = self.dupsample(x1)
        x1 = self._softmax(x1)
        x1 = self.split_v2(x1)
    
        x2 = self.head2(c2, c3, c4)
        x2 = self.dupsample2(x2)
        x2 = self._softmax(x2)
        x = self.add(x1, x2)
        x = P.ResizeNearestNeighbor((size[2], size[3]), False)(x)
        return x

    def construct_8_14_to_8(self, x):
        size = self.shape(x)
        _, c2, c3, c4 = self.encoder(x)
        x1 = self.head(c2, c3, c4)
        x1 = self.dupsample(x1)
        x1 = self._softmax(x1)
        x2 = self.head2(c2, c3, c4)
        x2 = self.dupsample2(x2)
        x2 = self._softmax(x2)
        x2 = self._merge(x2)
        x = self.add(x1, x2)
        x = P.ResizeNearestNeighbor((size[2], size[3]), False)(x)
        return x

    def construct_8_to_8(self, x):
        size = self.shape(x)
        _, c2, c3, c4 = self.encoder(x)
        x = self.head(c2, c3, c4)
        x = self.dupsample(x)
        x = P.ResizeNearestNeighbor((size[2], size[3]), False)(x)
        x = self._softmax(x)
        return x

    def construct_14_to_8(self, x):
        size = self.shape(x)
        _, c2, c3, c4 = self.encoder(x)
        x2 = self.head2(c2, c3, c4)
        x2 = self.dupsample2(x2)
        x2 = P.ResizeNearestNeighbor((size[2], size[3]), False)(x2)
        x2 = self._softmax(x2)
        x2 = self._merge(x2)
        return x2

    def construct_14_to_14(self, x):
        size = self.shape(x)
        _, c2, c3, c4 = self.encoder(x)
        x2 = self.head2(c2, c3, c4)
        x2 = self.dupsample2(x2)
        x2 = P.ResizeNearestNeighbor((size[2], size[3]), False)(x2)
        x2 = self._softmax(x2)
        return x2

    def test_construct(self, x):
        if self.mode == "01":
            return self.construct_8_to_8(x)
        elif self.mode == "02":
            return self.construct_14_to_8(x)
        elif self.mode == "03":
            return self.construct_8_14_to_8(x)
        elif self.mode == "04":
            return self.construct_14_to_14(x)
        else:
            return self.construct_8_14_to_14(x)

    def construct(self, x):
        if self.training:
            return self.train_construct(x)
        else:
            return self.test_construct(x)
    

class FeatureFused(nn.Cell):
    """Module for fused features"""

    def __init__(self, inter_channels=48):
        super(FeatureFused, self).__init__()
        self.conv2 = nn.SequentialCell(
            nn.Conv2d(512, inter_channels, 1, has_bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU()
        )
        self.conv3 = nn.SequentialCell(
            nn.Conv2d(1024, inter_channels, 1, has_bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU()
        )
        self.concat = P.Concat(axis=1)
        self.shape = P.Shape()


    def construct(self, c2, c3, c4):
        size = self.shape(c4)
        c2 = P.ResizeNearestNeighbor((size[2], size[3]), False)(c2)
        c2 = self.conv2(c2)
        c3 = P.ResizeNearestNeighbor((size[2], size[3]), False)(c3)
        c3 = self.conv3(c3)
        fused_feature = self.concat((c4, c3))
        fused_feature = self.concat((fused_feature, c2))
        return fused_feature


class DUHead(nn.Cell):
    def __init__(self, in_channels):
        super(DUHead, self).__init__()
        self.fuse = FeatureFused()
        self.block = nn.SequentialCell(
            nn.Conv2d(in_channels, 256, 3, pad_mode='same', has_bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, pad_mode='same', has_bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

    def construct(self, c2, c3, c4):
        fused_feature = self.fuse(c2, c3, c4)
        out = self.block(fused_feature)
        return out


class DUpsampling(nn.Cell):
    """DUsampling module"""

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(DUpsampling, self).__init__()
        self.scale_factor = scale_factor
        self.conv_w = nn.Conv2d(in_channels, out_channels * scale_factor * scale_factor, 1, has_bias=False)
        self.shape = P.Shape()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        
    def construct(self, x):
        x = self.conv_w(x)
        size = self.shape(x)
        n, c, h, w = size[0], size[1], size[2], size[3]
        
        # N, C, H, W --> N, W, H, C
        x = self.transpose(x, (0, 3, 2, 1))
        # N, W, H, C --> N, W, H * scale, C // scale
        x = self.reshape(x, (n, w, h * self.scale_factor, c // self.scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = self.transpose(x, (0, 2, 1, 3))
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = self.reshape(x, (n, h * self.scale_factor, w * self.scale_factor, c // (self.scale_factor * self.scale_factor)))
        # N, H * scale, W * scale, C // (scale ** 2) -- > N, C // (scale ** 2), H * scale, W * scale
        x = self.transpose(x, (0, 3, 1, 2))

        return x
