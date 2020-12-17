from src.nets import deeplabv3plus
from src.nets import deeplabv3plusv2
from src.nets.backbones import ofav100
from src.nets.backbones import resnext
from src.nets import dunetv2
nets_map = {'deeplabv3plus': deeplabv3plus.DeepLabV3Plus,
            'deeplabv3plusv2': deeplabv3plusv2.DeepLabV3PlusV2,
            'ofav100': ofav100.ofa_v100_gpu64_6ms,
            'dunetv2': dunetv2.DUNetV2}

backbones_map = {'resnext50': resnext.resnext50_32x4d,
                 'resnext101': resnext.resnext101_32x8d,
                 'ofav100': ofav100.ofa_v100_gpu64_6ms}
