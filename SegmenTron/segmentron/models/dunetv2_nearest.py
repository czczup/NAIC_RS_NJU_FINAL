"""Decoders Matter for Semantic Segmentation"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .segbase import SegBaseModel
from .model_zoo import MODEL_REGISTRY
from .fcn import _FCNHead
from ..config import cfg

__all__ = ['DUNetV2Nearest']


@MODEL_REGISTRY.register()
class DUNetV2Nearest(SegBaseModel):
    """Decoders Matter for Semantic Segmentation
    Reference:
        Zhi Tian, Tong He, Chunhua Shen, and Youliang Yan.
        "Decoders Matter for Semantic Segmentation:
        Data-Dependent Decoding Enables Flexible Feature Aggregation." CVPR, 2019
    """

    def __init__(self):
        super(DUNetV2Nearest, self).__init__()
        self.head = _DUHead(2144, norm_layer=self.norm_layer)
        self.dupsample = DUpsampling(256, 8, scale_factor=2)
        self.head2 = _DUHead(2144, norm_layer=self.norm_layer)
        self.dupsample2 = DUpsampling(256, 14, scale_factor=2)

        if self.aux:
            self.auxlayer = _FCNHead(1024, 256, norm_layer=self.norm_layer)
            self.aux_dupsample = DUpsampling(256, 8, scale_factor=2)
            self.auxlayer2 = _FCNHead(1024, 256, norm_layer=self.norm_layer)
            self.aux_dupsample2 = DUpsampling(256, 14, scale_factor=2)
        
        self.supervise_size = int(cfg.TRAIN.SUPERVISE_SIZE)

        self.__setattr__('decoder', ['head', 'dupsample', 'head2', 'dupsample2', 'auxlayer', 'aux_dupsample', 'auxlayer2', 'aux_dupsample2']
                                    if self.aux else ['dupsample', 'dupsample2', 'head'])
        
    def forward(self, x):
        size = (self.supervise_size, self.supervise_size)
        _, c2, c3, c4 = self.encoder(x)
        
        outputs1, outputs2 = list(), list()
        x1 = self.head(c2, c3, c4)
        x1 = self.dupsample(x1)
        
        x2 = self.head2(c2, c3, c4)
        x2 = self.dupsample2(x2)

        x1 = F.interpolate(x1, size, mode='nearest')
        x2 = F.interpolate(x2, size, mode='nearest')
        outputs1.append(x1)
        outputs2.append(x2)

        if self.aux and self.training:
            auxout1 = self.auxlayer(c3)
            auxout1 = self.aux_dupsample(auxout1)
            auxout1 = F.interpolate(auxout1, size, mode='nearest')

            auxout2 = self.auxlayer2(c3)
            auxout2 = self.aux_dupsample2(auxout2)
            auxout2 = F.interpolate(auxout2, size, mode='nearest')

            outputs1.append(auxout1)
            outputs2.append(auxout2)

        return tuple([outputs1, outputs2])  # 8/14

    def split_v2(self, x):
        x = torch.split(x, 1, dim=1)
        x = [x[0], x[1], x[2], x[7], x[7],
             x[3], x[3], x[4], x[4],
             x[5], x[5], x[6], x[6], x[7]]
        input = torch.cat(x, dim=1)
        return input

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
        c2 = self.conv2(F.interpolate(c2, size, mode='nearest'))
        c3 = self.conv3(F.interpolate(c3, size, mode='nearest'))
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
