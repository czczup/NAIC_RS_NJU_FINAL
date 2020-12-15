import torch
import torch.nn as nn
import torch.nn.functional as F


class DUNet(nn.Module):
    """Decoders Matter for Semantic Segmentation
    Reference:
        Zhi Tian, Tong He, Chunhua Shen, and Youliang Yan.
        "Decoders Matter for Semantic Segmentation:
        Data-Dependent Decoding Enables Flexible Feature Aggregation." CVPR, 2019
    """
    
    def __init__(self, nclass=[8, 14], get_backbone=None):
        super(DUNet, self).__init__()
        self.nclass = nclass
        self.encoder = get_backbone()
        self.norm_layer = nn.BatchNorm2d
        self.head = _DUHead(2144, norm_layer=self.norm_layer)
        self.head2 = _DUHead(2144, norm_layer=self.norm_layer)
        self.dupsample = DUpsampling(256, 8, scale_factor=2)
        self.dupsample2 = DUpsampling(256, 14, scale_factor=2)
        
        self.__setattr__('decoder', ['dupsample', 'dupsample2', 'head', 'head2'])
    
    def merge(self, x):
        x = torch.split(x, 1, dim=1)
        x = [x[0], x[1], x[2], torch.add(x[5], x[6]),
             torch.add(x[7], x[8]), torch.add(x[9], x[10]),
             torch.add(x[11], x[12]),
             torch.add(torch.add(x[3], x[4]), x[13])]
        input = torch.cat(x, dim=1)
        return input
    
    def split(self, x):
        x = torch.split(x, 1, dim=1)
        x = [x[0], x[1], x[2], x[7] / 3.0, x[7] / 3.0,
             x[3] / 2.0, x[3] / 2.0, x[4] / 2.0, x[4] / 2.0,
             x[5] / 2.0, x[5] / 2.0, x[6] / 2.0, x[6] / 2.0,
             x[7] / 3.0]
        input = torch.cat(x, dim=1)
        return input
    
    def split_v2(self, x):
        x = torch.split(x, 1, dim=1)
        x = [x[0], x[1], x[2], x[7], x[7],
             x[3], x[3], x[4], x[4],
             x[5], x[5], x[6], x[6], x[7]]
        input = torch.cat(x, dim=1)
        return input
    
    def forward_8_to_8(self, x):
        _, c2, c3, c4 = self.encoder(x)
        outputs = list()
        x = self.head(c2, c3, c4)  # 256
        x = self.dupsample(x)  # 8
        x = F.softmax(x, dim=1)
        outputs.append(x)
        return tuple(outputs)
    
    def forward_14_to_14(self, x):
        _, c2, c3, c4 = self.encoder(x)
        outputs = list()
        x = self.head2(c2, c3, c4)  # 256
        x = self.dupsample2(x)  # 14
        x = F.softmax(x, dim=1)
        outputs.append(x)
        return tuple(outputs)
    
    def forward_14_to_8(self, x):
        _, c2, c3, c4 = self.encoder(x)
        outputs = list()
        x = self.head2(c2, c3, c4)  # 256
        x = self.dupsample2(x)  # 14
        x = F.softmax(x, dim=1)
        x = self.merge(x)  # 8
        outputs.append(x)
        return tuple(outputs)
    
    def forward_8_14_to_8(self, x):
        c_, c2, c3, c4 = self.encoder(x)
        outputs = list()
        x1 = self.head(c2, c3, c4)  # 256
        x1 = self.dupsample(x1)  # 8
        x1 = F.softmax(x1, dim=1)  # 8
        
        x2 = self.head2(c2, c3, c4)  # 256
        x2 = self.dupsample2(x2)  # 14
        x2 = F.softmax(x2, dim=1)  # 14
        x2 = self.merge(x2)  # 8
        
        x = x1 + x2
        outputs.append(x)
        return tuple(outputs)
    
    def forward_8_14_to_14(self, x):
        c_, c2, c3, c4 = self.encoder(x)
        outputs = list()
        x1 = self.head(c2, c3, c4)  # 256
        x1 = self.dupsample(x1)  # 8
        x1 = F.softmax(x1, dim=1)  # 8
        x1 = self.split(x1)  # 14
        
        x2 = self.head2(c2, c3, c4)  # 256
        x2 = self.dupsample2(x2)  # 14
        x2 = F.softmax(x2, dim=1)  # 14
        x = x1 + x2
        outputs.append(x)
        return tuple(outputs)
    
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
    
    def forward(self, x, mode):
        if mode == "01":
            return self.forward_8_to_8(x)
        elif mode == "02":
            return self.forward_14_to_8(x)
        elif mode == "03":
            return self.forward_8_14_to_8(x)
        elif mode == "04":
            return self.forward_14_to_14(x)
        elif mode == "05":
            return self.forward_8_14_to_14(x)
        elif mode == "06":
            return self.forward_8_14_to_14_v2(x)


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
