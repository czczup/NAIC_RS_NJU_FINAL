from .build import BACKBONE_REGISTRY, get_segmentation_backbone
from .xception import *
from .mobilenet import *
from .resnet import *
from .hrnet import *
from .eespnet import *
from .resnext import *
from .ibn_resnext import *
from .ibn_res2net import *
from .ibn_resnet import *
from .shufflenetv2p import *
from .ibn_shufflenetv2p import *
from .ofa_1080ti_gpu64_27ms.main import *
from .tiny_ofa_1080ti_gpu64_27ms.main import *
from .ofa_note10_lat_8ms.main import *
from .ofa_v100_gpu64_6ms.main import *
from .ofa_flops_595m.main import *
from .vovnet import *
