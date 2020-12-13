import torch
from deeplabv3plus import DeepLabV3Plus
from dunet import DUNet
from ibn_resnet import ibn_resnet50
from ibn_resnext import ibn_b_resnext50_32x4d
from resnext import resnext50_32x4d
from segmentron.models.backbones.ofa_1080ti_gpu64_27ms.main import ofa_1080ti_gpu64_27ms
from segmentron.models.backbones.tiny_ofa_1080ti_gpu64_27ms.main import tiny_ofa_1080ti_gpu64_27ms, ibn_tiny_ofa_1080ti_gpu64_27ms
from segmentron.models.backbones.shufflenetv2p import shufflenetv2_plus
from segmentron.models.backbones.ibn_shufflenetv2p import ibn_shufflenetv2_plus
from segmentron.models.backbones.ofa_note10_lat_8ms.main import ofa_note10_lat_8ms, ibn_ofa_note10_lat_8ms
from segmentron.models.backbones.ofa_v100_gpu64_6ms.main import ofa_v100_gpu64_6ms, ibn_ofa_v100_gpu64_6ms
from segmentron.models.backbones.resnet import resnet50
from segmentron.models.backbones.vovnet import vovnet19_dw
from deeplabv3plus_nearest import DeepLabV3PlusNearest
from tools import fuse_module
try:
    import apex
except:
    print("apex is not installed successfully")


def init_model(args):
    device = torch.device("cuda")
    
    # model = DeepLabV3Plus(nclass=[8,14], get_backbone=ofa_1080ti_gpu64_27ms, channels=[32, 1664])
    # model = DeepLabV3Plus(nclass=[8,14], get_backbone=shufflenetv2_plus, channels=[68,1280])
    # model = DeepLabV3Plus(nclass=[8,14], get_backbone=ibn_resnet50, channels=[256, 2048])
    # model = DeepLabV3Plus(nclass=[8,14], get_backbone=ofa_note10_lat_8ms, channels=[24, 160])
    # model = DeepLabV3Plus(nclass=[8,14], get_backbone=resnext50_32x4d, channels=[256, 2048])
    # model = DeepLabV3Plus(nclass=[8,14], get_backbone=tiny_ofa_1080ti_gpu64_27ms, channels=[32, 416])
    model = DeepLabV3Plus(nclass=[8,14], get_backbone=ofa_v100_gpu64_6ms, channels=[32, 248])
    # model = DeepLabV3PlusNearest(nclass=[8,14], get_backbone=ofa_v100_gpu64_6ms, channels=[32, 248])
    # model = DeepLabV3Plus(nclass=[8,14], get_backbone=vovnet19_dw, channels=[256, 1024])

    model_path = "../model/%s" % args.model
    print('load test model from {}'.format(model_path))
    
    if args.quantize:
        model = fuse_module(model)
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        state_dict_ = dict()
        for k, v in state_dict.items():
            if v.dtype == torch.qint8:
                state_dict_[k] = v.dequantize()
            else:
                state_dict_[k] = v
        msg = model.load_state_dict(state_dict_, strict=False)
    else:
        msg = model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage), strict=False)
        model = fuse_module(model)
    
    print(msg)
    model.eval()
    model = model.to(device)

    try:
        model = apex.amp.initialize(model.cuda(), opt_level="O3")
    except:
        pass
    
    return model
