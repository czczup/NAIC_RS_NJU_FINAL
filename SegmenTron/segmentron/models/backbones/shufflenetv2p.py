import torch
import torch.nn as nn
from .build import BACKBONE_REGISTRY
from ...config import cfg

__all__ = ['shufflenetv2_plus']

class SELayer(nn.Module):

    def __init__(self, inplanes, isTensor=True):
        super(SELayer, self).__init__()
        if isTensor:
            # if the input is (N, C, H, W)
            self.SE_opr = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(inplanes, inplanes // 4, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(inplanes // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes // 4, inplanes, kernel_size=1, stride=1, bias=False),
            )
        else:
            # if the input is (N, C)
            self.SE_opr = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Linear(inplanes, inplanes // 4, bias=False),
                nn.BatchNorm1d(inplanes // 4),
                nn.ReLU(inplace=True),
                nn.Linear(inplanes // 4, inplanes, bias=False),
            )

    def forward(self, x):
        atten = self.SE_opr(x)
        atten = torch.clamp(atten + 3, 0, 6) / 6
        return x * atten


class HS(nn.Module):

    def __init__(self):
        super(HS, self).__init__()

    def forward(self, inputs):
        clip = torch.clamp(inputs + 3, 0, 6) / 6
        return inputs * clip



class Shufflenet(nn.Module):

    def __init__(self, inp, oup, base_mid_channels, *, ksize, stride, activation, useSE, mode):
        super(Shufflenet, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        assert ksize in [3, 5, 7]
        assert base_mid_channels == oup//2

        self.base_mid_channel = base_mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp
        self.mode = mode
        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, base_mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(base_mid_channels),
            None,
            # dw
            nn.Conv2d(base_mid_channels, base_mid_channels, ksize, stride, pad, groups=base_mid_channels, bias=False),
            nn.BatchNorm2d(base_mid_channels),
            # pw-linear
            nn.Conv2d(base_mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            None,
        ]
        if activation == 'ReLU':
            assert useSE == False
            '''This model should not have SE with ReLU'''
            branch_main[2] = nn.ReLU(inplace=True)
            branch_main[-1] = nn.ReLU(inplace=True)
        else:
            branch_main[2] = HS()
            branch_main[-1] = HS()
            if useSE:
                branch_main.append(SELayer(outputs))
        self.branch_main = nn.Sequential(*branch_main)

        if mode == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                None,
            ]
            if activation == 'ReLU':
                branch_proj[-1] = nn.ReLU(inplace=True)
            else:
                branch_proj[-1] = HS()
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.mode == 1:
            x_proj, x = channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.mode == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

class Shuffle_Xception(nn.Module):

    def __init__(self, inp, oup, base_mid_channels, *, stride, activation, useSE, mode):
        super(Shuffle_Xception, self).__init__()

        assert stride in [1, 2]
        assert base_mid_channels == oup//2

        self.base_mid_channel = base_mid_channels
        self.stride = stride
        self.ksize = 3
        self.pad = 1
        self.inp = inp
        self.mode = mode
        outputs = oup - inp

        branch_main = [
            # dw
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            # pw
            nn.Conv2d(inp, base_mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(base_mid_channels),
            None,
            # dw
            nn.Conv2d(base_mid_channels, base_mid_channels, 3, stride, 1, groups=base_mid_channels, bias=False),
            nn.BatchNorm2d(base_mid_channels),
            # pw
            nn.Conv2d(base_mid_channels, base_mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(base_mid_channels),
            None,
            # dw
            nn.Conv2d(base_mid_channels, base_mid_channels, 3, stride, 1, groups=base_mid_channels, bias=False),
            nn.BatchNorm2d(base_mid_channels),
            # pw
            nn.Conv2d(base_mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            None,
        ]

        if activation == 'ReLU':
            branch_main[4] = nn.ReLU(inplace=True)
            branch_main[9] = nn.ReLU(inplace=True)
            branch_main[14] = nn.ReLU(inplace=True)
        else:
            branch_main[4] = HS()
            branch_main[9] = HS()
            branch_main[14] = HS()
        assert None not in branch_main

        if useSE:
            assert activation != 'ReLU'
            branch_main.append(SELayer(outputs))

        self.branch_main = nn.Sequential(*branch_main)

        if self.mode == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                None,
            ]
            if activation == 'ReLU':
                branch_proj[-1] = nn.ReLU(inplace=True)
            else:
                branch_proj[-1] = HS()
            self.branch_proj = nn.Sequential(*branch_proj)

    def forward(self, old_x):
        if self.mode == 1:
            x_proj, x = channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.mode == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)


def channel_shuffle(x):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]


class ShuffleNetV2_Plus(nn.Module):
    def __init__(self, input_size=224, n_class=1000, architecture=None, model_size='Large'):
        super(ShuffleNetV2_Plus, self).__init__()

        assert input_size % 32 == 0
        assert architecture is not None

        self.stage_repeats = [4, 4, 8, 4]
        if model_size == 'Large':
            self.stage_out_channels = [-1, 16, 68, 168, 336, 672, 1280]
        elif model_size == 'Medium':
            self.stage_out_channels = [-1, 16, 48, 128, 256, 512, 1280]
        elif model_size == 'Small':
            self.stage_out_channels = [-1, 16, 36, 104, 208, 416, 1280]
        else:
            raise NotImplementedError

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            HS(),
        )
        
        self.stages = []

        archIndex = 0
        for idxstage in range(len(self.stage_repeats)):
            self.features = []
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]

            activation = 'HS' if idxstage >= 1 else 'ReLU'
            useSE = 'True' if idxstage >= 2 else False
            for i in range(numrepeat):
                if i == 0:
                    inp, outp, stride, mode = input_channel, output_channel, 2, 2
                else:
                    inp, outp, stride, mode = input_channel // 2, output_channel, 1, 1
                
                if idxstage == 2 or idxstage == 3:
                    stride = 1
                
                blockIndex = architecture[archIndex]
                archIndex += 1
                if blockIndex == 0:
                    self.features.append(Shufflenet(inp, outp, base_mid_channels=outp // 2, ksize=3, stride=stride,
                                    activation=activation, useSE=useSE, mode=mode))
                elif blockIndex == 1:
                    self.features.append(Shufflenet(inp, outp, base_mid_channels=outp // 2, ksize=5, stride=stride,
                                    activation=activation, useSE=useSE, mode=mode))
                elif blockIndex == 2:
                    self.features.append(Shufflenet(inp, outp, base_mid_channels=outp // 2, ksize=7, stride=stride,
                                    activation=activation, useSE=useSE, mode=mode))
                elif blockIndex == 3:
                    self.features.append(Shuffle_Xception(inp, outp, base_mid_channels=outp // 2, stride=stride,
                                    activation=activation, useSE=useSE, mode=mode))
                else:
                    raise NotImplementedError
                input_channel = output_channel
            
            if idxstage == 0:
                self.stage1 = nn.Sequential(*self.features)
            elif idxstage == 1:
                self.stage2 = nn.Sequential(*self.features)
            elif idxstage == 2:
                self.stage3 = nn.Sequential(*self.features)
            elif idxstage == 3:
                self.stage4 = nn.Sequential(*self.features)
        assert archIndex == len(architecture)

        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channel, 1280, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1280),
            HS()
        )
        self._initialize_weights()

    def forward(self, x):
        outs = list()
        x = self.first_conv(x)
        x = self.stage1(x)
        outs.append(x)
        x = self.stage2(x)
        outs.append(x)
        x = self.stage3(x)
        outs.append(x)
        x = self.stage4(x)
        x = self.conv_last(x)
        outs.append(x)
        return outs

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name or 'SE' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


@BACKBONE_REGISTRY.register()
def shufflenetv2_plus(norm_layer=nn.BatchNorm2d):
    architecture = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
    return ShuffleNetV2_Plus(architecture=architecture)


if __name__ == "__main__":
    architecture = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
    model = ShuffleNetV2_Plus(architecture=architecture)
    # for name in model.state_dict():
    #     print(name)
    # print(model)
    # state_dict = torch.load("ShuffleNetV2+.Large.pth.tar")['state_dict']
    # translater = open("shufflenetv2p.csv", "r")
    # content = [item[:-1].split(",") for item in translater.readlines()]
    # old2new = {item[1]:item[0] for item in content}
    # print(old2new)
    # state_dict = {old2new[k]:v for k, v in state_dict.items() if k in old2new}
    # model.load_state_dict(state_dict)
    
    test_data = torch.rand(2, 3, 640, 640)
    test_outputs = model(test_data)
    for out in test_outputs:
        print(out.shape)