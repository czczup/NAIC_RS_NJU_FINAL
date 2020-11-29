import torch
import torch.nn as nn
import torch.nn.functional as F

from .segbase import SegBaseModel
from .model_zoo import MODEL_REGISTRY
from ..modules import _ConvBNReLU, SeparableConv2d, _ASPP, _FCNHead
from ..config import cfg

__all__ = ['DeepLabV3PlusV3']


@MODEL_REGISTRY.register(name='DeepLabV3_Plus_V3')
class DeepLabV3PlusV3(SegBaseModel):
    r"""DeepLabV3Plus
    Reference:
        Chen, Liang-Chieh, et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic
        Image Segmentation."
    """
    def __init__(self):
        super(DeepLabV3PlusV3, self).__init__()
        c1_channels = 256
        c3_channels = 1024
        c4_channels = 2048
        datasetA_nclass = 8
        datasetC_nclass = 14
        self.head = _DeepLabHead(datasetA_nclass, c1_channels=c1_channels, c4_channels=c4_channels)
        self.head2 = _DeepLabHead(datasetC_nclass, c1_channels=c1_channels, c4_channels=c4_channels)
        self.tiny_backbone = _TinyBackbone(norm_layer=IBN)
        self.supervise_size = int(cfg.TRAIN.SUPERVISE_SIZE)
        
        if self.aux:
            self.auxlayer = _FCNHead(c3_channels, datasetA_nclass)
            self.auxlayer2 = _FCNHead(c3_channels, datasetC_nclass)

        self.__setattr__('decoder', ['head', 'auxlayer', 'head2', 'auxlayer2'] if self.aux else ['head', 'head2'])

    def forward(self, x):
        size = (self.supervise_size, self.supervise_size)
        c1, _, c3, c4 = self.encoder(x)
        tiny_out = self.tiny_backbone(x)
        outputs1 = list()
        x = self.head(c4, c1, tiny_out)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs1.append(x)
        if self.aux and self.training:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs1.append(auxout)

        outputs2 = list()
        x = self.head2(c4, c1, tiny_out)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs2.append(x)
        if self.aux and self.training:
            auxout = self.auxlayer2(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs2.append(auxout)
        return tuple([outputs1, outputs2])
    

class _DeepLabHead(nn.Module):
    def __init__(self, nclass, c1_channels=256, c4_channels=2048, norm_layer=nn.BatchNorm2d):
        super(_DeepLabHead, self).__init__()
        self.aspp = _ASPP(c4_channels, 256)
        last_channels = 256
        self.c1_block = _ConvBNReLU(c1_channels, 48, 1, norm_layer=norm_layer)
        last_channels += 48
        self.tiny_block = _ConvBNReLU(c1_channels, 48, 1, norm_layer=norm_layer)
        last_channels += 48
        self.block = nn.Sequential(
            SeparableConv2d(last_channels, 256, 3, norm_layer=norm_layer, relu_first=False),
            SeparableConv2d(256, 256, 3, norm_layer=norm_layer, relu_first=False),
            nn.Conv2d(256, nclass, 1))

    def forward(self, x, c1, tiny_out):
        c1 = F.interpolate(c1, (160, 160), mode='bilinear', align_corners=True) # (256, 160, 160)
        x = F.interpolate(x, (80, 80), mode='bilinear', align_corners=True)
        x = self.aspp(x) # (2048, 80, 80) -> (256, 80, 80)
        x = F.interpolate(x, (160, 160), mode='bilinear', align_corners=True)
        c1 = self.c1_block(c1) # (48, 160, 160)
        tiny_out = self.tiny_block(tiny_out) # (48, 160, 160)
        return self.block(torch.cat([x, c1, tiny_out], dim=1))


class IBN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`
    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """
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


class _TinyBackbone(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(_TinyBackbone, self).__init__()
        self.block1 = _ConvBNReLU(3, 64, 3, stride=2, padding=1, norm_layer=norm_layer)
        self.block2 = _ConvBNReLU(64, 128, 3, stride=2, padding=1, norm_layer=norm_layer)
        self.block3 = _ConvBNReLU(128, 256, 3, stride=1, padding=1, norm_layer=norm_layer)
    
    def forward(self, x):
        x = F.interpolate(x, (640, 640), mode='bilinear', align_corners=True)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

