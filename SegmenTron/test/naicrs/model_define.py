import torch
from deeplabv3plus import DeepLabV3Plus
from dunet import DUNet
from ibn_resnet import ibn_resnet50
from ibn_resnext import ibn_b_resnext50_32x4d
from resnext import resnext50_32x4d
from segmentron.models.backbones.ofa_1080ti_gpu64_27ms.main import ofa_1080ti_gpu64_27ms
from segmentron.models.backbones.tiny_ofa_1080ti_gpu64_27ms.main import tiny_ofa_1080ti_gpu64_27ms
from segmentron.models.backbones.shufflenetv2p import shufflenetv2_plus
from segmentron.models.backbones.ibn_shufflenetv2p import ibn_shufflenetv2_plus

try:
    import apex
except:
    print("apex is not installed successfully")


def init_model(model_filename):
    device = torch.device("cuda")
    model = DeepLabV3Plus(nclass=[8,14], get_backbone=ofa_1080ti_gpu64_27ms, channels=[32, 1664])
    # model = DeepLabV3Plus(nclass=[8,14], get_backbone=tiny_ofa_1080ti_gpu64_27ms, channels=[32, 416])
    # model = DeepLabV3Plus(nclass=[8,14], get_backbone=shufflenetv2_plus, channels=[68,1280])
    # model = DeepLabV3Plus(nclass=[8,14], get_backbone=ibn_resnet50, channels=[256, 2048])

    model_path = "../model/%s" % model_filename
    print('load test model from {}'.format(model_path))
    msg = model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage), strict=False)
    print(msg)
    model.eval()
    model = model.to(device)

    try:
        model = apex.amp.initialize(model.cuda(), opt_level="O1")
    except:
        pass
    
    return model
