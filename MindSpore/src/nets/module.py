import mindspore.nn as nn
from mindspore.ops import operations as P
from collections import OrderedDict



class ConvBNReLU(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, group=1, use_batch_statistics=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding,
                              dilation=dilation, group=group, has_bias=False)
        self.bn = nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics)
        self.relu = nn.ReLU()
    
    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ASPP(nn.Cell):
    def __init__(self, in_channels, out_channels, output_stride=8, use_batch_statistics=True):
        super(ASPP, self).__init__()
        
        if output_stride == 16:
            dilations = [6, 12, 18]
        elif output_stride == 8:
            dilations = [12, 24, 36]
        elif output_stride == 32:
            dilations = [6, 12, 18]
        else:
            raise NotImplementedError
        
        self.aspp0 = nn.SequentialCell([
            nn.Conv2d(in_channels, out_channels, kernel_size=1, has_bias=False, weight_init='he_uniform'),
            nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics),
            nn.ReLU()
        ])
        self.aspp1 = SeparableConv2d(in_channels, out_channels, dilation=dilations[0],
                                     use_batch_statistics=use_batch_statistics)
        self.aspp2 = SeparableConv2d(in_channels, out_channels, dilation=dilations[1],
                                     use_batch_statistics=use_batch_statistics)
        self.aspp3 = SeparableConv2d(in_channels, out_channels, dilation=dilations[2],
                                     use_batch_statistics=use_batch_statistics)
        
        self.image_pooling = nn.SequentialCell([
            nn.Conv2d(in_channels, out_channels, kernel_size=1, weight_init='he_uniform'),
            nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics),
            nn.ReLU()
        ])
        
        self.conv = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, weight_init='he_uniform')
        self.bn = nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        self.concat = P.Concat(axis=1)
        self.shape = P.Shape()
    
    def construct(self, x):
        size = self.shape(x)
        pool = nn.AvgPool2d(size[2])(x)
        pool = self.image_pooling(pool)
        pool = P.ResizeNearestNeighbor((size[2], size[3]), True)(pool)  # TODO: bilinear
        
        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        
        x = self.concat((pool, x0))
        x = self.concat((x, x1))
        x = self.concat((x, x2))
        x = self.concat((x, x3))
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        # x = self.dropout(x)
        return x


class SeparableConv2d(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, has_bias=False,
                 use_batch_statistics=True):
        super().__init__()
        depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                              stride=stride, pad_mode="same",
                              dilation=dilation, group=in_channels, has_bias=has_bias)
        bn_depth = nn.BatchNorm2d(in_channels, use_batch_statistics=use_batch_statistics)
        pointwise = nn.Conv2d(in_channels, out_channels, 1, has_bias=has_bias)
        bn_point = nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics)
        
        self.block = nn.SequentialCell(OrderedDict([('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('relu1', nn.ReLU()),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point),
                                                    ('relu2', nn.ReLU())
                                                    ]))
    
    def construct(self, x):
        return self.block(x)
    
    
class FCNHead(nn.Cell):
    def __init__(self, in_channels, channels, use_batch_statistics=True):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.SequentialCell([
            nn.Conv2d(in_channels, inter_channels, 3, pad_mode='same', has_bias=False),
            nn.BatchNorm2d(inter_channels, use_batch_statistics=use_batch_statistics),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1, has_bias=True)
        ])

    def construct(self, x):
        return self.block(x)