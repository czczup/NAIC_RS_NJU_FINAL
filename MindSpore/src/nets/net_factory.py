from src.nets import deeplabv3plus
from src.nets import deeplabv3plusv2
from src.nets import ofav100
nets_map = {'deeplabv3plus': deeplabv3plus.DeepLabV3Plus,
            'deeplabv3plusv2': deeplabv3plusv2.DeepLabV3PlusV2,
            'ofav100': ofav100.ofa_v100_gpu64_6ms}


