import torch
import torch.nn as nn
import torch.nn.functional as F

from .segbase import SegBaseModel
from .model_zoo import MODEL_REGISTRY
from ..modules import _ConvBNReLU, SeparableConv2d, _ASPP, _FCNHead
from ..config import cfg
from .backbones import get_segmentation_backbone

__all__ = ['DeepLabV3PlusV2Teacher']


@MODEL_REGISTRY.register(name='DeepLabV3_Plus_V2_Teacher')
class DeepLabV3PlusV2Teacher(SegBaseModel):
    r"""DeepLabV3Plus
    Reference:
        Chen, Liang-Chieh, et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic
        Image Segmentation."
    """
    def __init__(self):
        super(DeepLabV3PlusV2Teacher, self).__init__(need_backbone=False)
        self.get_backbone()
        c1_channels = 256
        c3_channels = 1024
        c4_channels = 2048
        datasetA_nclass = 8
        datasetC_nclass = 14
        self.head = _DeepLabHead(datasetA_nclass, c1_channels=c1_channels, c4_channels=c4_channels)
        self.head2 = _DeepLabHead(datasetC_nclass, c1_channels=c1_channels, c4_channels=c4_channels)
        self.supervise_size = int(cfg.TRAIN.SUPERVISE_SIZE)
        
        if self.aux:
            self.auxlayer = _FCNHead(c3_channels, datasetA_nclass)
            self.auxlayer2 = _FCNHead(c3_channels, datasetC_nclass)

        self.__setattr__('decoder', ['head', 'auxlayer', 'head2', 'auxlayer2'] if self.aux else ['head', 'head2'])

    def get_backbone(self):
        self.backbone = cfg.DISTILL.TEACHER_BACKBONE.lower()
        self.encoder = get_segmentation_backbone(self.backbone, self.norm_layer, pretrain=False)
        
    def forward(self, x):
        size = (self.supervise_size, self.supervise_size)
        c1, _, c3, c4 = self.encoder(x)
        
        x, aspp1 = self.head(c4, c1)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs1 = list([x]) # 8
        
        x, aspp2 = self.head2(c4, c1)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs2 = list([x]) # 14
        aspps = list([aspp1, aspp2])
        
        return tuple([outputs1, outputs2, aspps])
    

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
        aspp = x
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        c1 = self.c1_block(c1)
        return self.block(torch.cat([x, c1], dim=1)), aspp

