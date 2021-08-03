from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .build import BACKBONE_REGISTRY

__all__ = ["VoVNet", "vovnet19", "vovnet19_slim", "vovnet19_dw", "vovnet19_slim_dw"]

def get_same_pad(ksize, dilation):
    pad = ((ksize - 1) * dilation) / 2
    return int(pad)

VoVNet19_slim_dw_eSE = {
    'stem': [64, 64, 64],
    'stage_conv_ch': [64, 80, 96, 112],
    'stage_out_ch': [112, 256, 384, 512],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "eSE": True,
    "dw" : True
}

VoVNet19_dw_eSE = {
    'stem': [64, 64, 64],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "eSE": True,
    "dw" : True
}

VoVNet19_slim_eSE = {
    'stem': [64, 64, 128],
    'stage_conv_ch': [64, 80, 96, 112],
    'stage_out_ch': [112, 256, 384, 512],
    'layer_per_block': 3,
    'block_per_stage': [1, 1, 1, 1],
    'eSE' : True,
    "dw" : False
}

VoVNet19_eSE = {
    'stem': [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "eSE": True,
    "dw" : False
}


_STAGE_SPECS = {
    "V-19-slim-dw-eSE": VoVNet19_slim_dw_eSE,
    "V-19-dw-eSE": VoVNet19_dw_eSE,
    "V-19-slim-eSE": VoVNet19_slim_eSE,
    "V-19-eSE": VoVNet19_eSE,
}

def dw_conv3x3(in_channels, out_channels, module_name, postfix,
            stride=1, kernel_size=3, padding=1, dilation=1):
    """3x3 convolution with padding"""
    return [
        ('{}_{}/dw_conv3x3'.format(module_name, postfix),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=get_same_pad(ksize=3, dilation=dilation),
                      groups=out_channels,
                      dilation=dilation,
                      bias=False)),
        ('{}_{}/pw_conv1x1'.format(module_name, postfix),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      groups=1,
                      bias=False)),
        ('{}_{}/pw_norm'.format(module_name, postfix), nn.BatchNorm2d(out_channels)),
        ('{}_{}/pw_relu'.format(module_name, postfix), nn.ReLU(inplace=True)),
    ]

def conv3x3(
    in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=3, padding=1, dilation=1
):
    """3x3 convolution with padding"""
    return [
        (
            f"{module_name}_{postfix}/conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=get_same_pad(ksize=3, dilation=dilation),
                groups=groups,
                dilation=dilation,
                bias=False,
            ),
        ),
        (f"{module_name}_{postfix}/norm", nn.BatchNorm2d(out_channels)),
        (f"{module_name}_{postfix}/relu", nn.ReLU(inplace=True)),
    ]


def conv1x1(
    in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=1, padding=0
):
    """1x1 convolution with padding"""
    return [
        (
            f"{module_name}_{postfix}/conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        ),
        (f"{module_name}_{postfix}/norm", nn.BatchNorm2d(out_channels)),
        (f"{module_name}_{postfix}/relu", nn.ReLU(inplace=True)),
    ]


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class eSEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(eSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        self.hsigmoid = Hsigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.hsigmoid(x)
        return input * x


class _OSA_module(nn.Module):
    def __init__(
        self, in_ch, stage_ch, concat_ch, layer_per_block, module_name, SE=False,
            identity=False, depthwise=False, dilation=1):

        super(_OSA_module, self).__init__()

        self.identity = identity
        self.depthwise = depthwise
        self.isReduced = False
        self.layers = nn.ModuleList()
        in_channel = in_ch
        if self.depthwise and in_channel != stage_ch:
            self.isReduced = True
            self.conv_reduction = nn.Sequential(
                OrderedDict(conv1x1(in_channel, stage_ch,
                  "{}_reduction".format(module_name), "0")))
        for i in range(layer_per_block):
            if self.depthwise:
                self.layers.append(
                    nn.Sequential(OrderedDict(dw_conv3x3(stage_ch, stage_ch, module_name, i, dilation=dilation))))
            else:
                self.layers.append(
                    nn.Sequential(OrderedDict(conv3x3(in_channel, stage_ch, module_name, i, dilation=dilation)))
                )
            in_channel = stage_ch

        # feature aggregation
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(
            OrderedDict(conv1x1(in_channel, concat_ch, module_name, "concat"))
        )

        self.ese = eSEModule(concat_ch)

    def forward(self, x):

        identity_feat = x

        output = []
        output.append(x)
        if self.depthwise and self.isReduced:
            x = self.conv_reduction(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        xt = self.concat(x)
        xt = self.ese(xt)

        if self.identity:
            xt = xt + identity_feat

        return xt


class _OSA_stage(nn.Sequential):
    def __init__(
        self,
        in_ch,
        stage_ch,
        concat_ch,
        block_per_stage,
        layer_per_block,
        stage_num, SE=False,
        depthwise=False):

        super(_OSA_stage, self).__init__()

        if stage_num == 3:
            self.add_module("Pooling", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        if block_per_stage != 1:
            SE = False
        module_name = f"OSA{stage_num}_1"
        if "OSA5" in module_name:
            self.add_module(
                module_name,
                _OSA_module(in_ch, stage_ch, concat_ch, layer_per_block, module_name, SE, depthwise=depthwise, dilation=4)
            )
        elif "OSA4" in module_name:
            self.add_module(
                module_name,
                _OSA_module(in_ch, stage_ch, concat_ch, layer_per_block, module_name, SE, depthwise=depthwise,
                            dilation=2)
            )
        else:
            self.add_module(
                module_name, _OSA_module(in_ch, stage_ch, concat_ch, layer_per_block, module_name, SE, depthwise=depthwise)
            )
            
        for i in range(block_per_stage - 1):
            if i != block_per_stage - 2:  # last block
                SE = False
            module_name = f"OSA{stage_num}_{i + 2}"
            self.add_module(
                module_name,
                _OSA_module(
                    concat_ch, stage_ch, concat_ch, layer_per_block, module_name, SE, identity=True, depthwise=depthwise
                ),
            )


class VoVNet(nn.Module):
    def __init__(self, conv_body, out_features=None):
        
        super(VoVNet, self).__init__()

        stage_specs = _STAGE_SPECS[conv_body]

        stem_ch = stage_specs["stem"]
        config_stage_ch = stage_specs["stage_conv_ch"]
        config_concat_ch = stage_specs["stage_out_ch"]
        block_per_stage = stage_specs["block_per_stage"]
        layer_per_block = stage_specs["layer_per_block"]
        SE = stage_specs["eSE"]
        depthwise = stage_specs["dw"]

        self._out_features = out_features

        # Stem module
        conv_type = dw_conv3x3 if depthwise else conv3x3
        stem = conv3x3(3, stem_ch[0], "stem", "1", 2)
        stem += conv_type(stem_ch[0], stem_ch[1], "stem", "2", 1)
        stem += conv_type(stem_ch[1], stem_ch[2], "stem", "3", 2)
        self.add_module("stem", nn.Sequential((OrderedDict(stem))))
        current_stirde = 4
        self._out_feature_strides = {"stem": current_stirde, "stage2": current_stirde}
        self._out_feature_channels = {"stem": stem_ch[2]}

        stem_out_ch = [stem_ch[2]]
        in_ch_list = stem_out_ch + config_concat_ch[:-1]
        # OSA stages
        self.stage_names = []
        for i in range(4):  # num_stages
            name = "stage%d" % (i + 2)  # stage 2 ... stage 5
            self.stage_names.append(name)
            self.add_module(
                name,
                _OSA_stage(
                    in_ch_list[i],
                    config_stage_ch[i],
                    config_concat_ch[i],
                    block_per_stage[i],
                    layer_per_block,
                    i + 2,
                    SE,
                    depthwise,
                ),
            )

            self._out_feature_channels[name] = config_concat_ch[i]
            if not i == 0:
                self._out_feature_strides[name] = current_stirde = int(current_stirde * 2)

        # initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for name in self.stage_names:
            x = getattr(self, name)(x)
            if name in self._out_features:
                outputs[name] = x
        
        return [outputs[stage] for stage in self._out_features]

@BACKBONE_REGISTRY.register()
def vovnet19(norm_layer=nn.BatchNorm2d):
    out_features = ["stage2", "stage3", "stage4", "stage5"]
    return VoVNet(conv_body="V-19-eSE", out_features=out_features)

@BACKBONE_REGISTRY.register()
def vovnet19_slim(norm_layer=nn.BatchNorm2d):
    out_features = ["stage2", "stage3", "stage4", "stage5"]
    return VoVNet(conv_body="V-19-slim-eSE", out_features=out_features)

@BACKBONE_REGISTRY.register()
def vovnet19_dw(norm_layer=nn.BatchNorm2d):
    out_features = ["stage2", "stage3", "stage4", "stage5"]
    return VoVNet(conv_body="V-19-dw-eSE", out_features=out_features)

@BACKBONE_REGISTRY.register()
def vovnet19_slim_dw(norm_layer=nn.BatchNorm2d):
    out_features = ["stage2", "stage3", "stage4", "stage5"]
    return VoVNet(conv_body="V-19-slim-dw-eSE", out_features=out_features)


if __name__ == '__main__':
    input = torch.rand(1, 3, 256, 256)
    model = vovnet19_slim_dw()
    outs = model(input)
    for key in ["stage2", "stage3", "stage4", "stage5"]:
        print(outs[key].shape)
    
