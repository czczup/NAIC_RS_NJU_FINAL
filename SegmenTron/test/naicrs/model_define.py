import torch
from deeplabv3plus import DeepLabV3Plus
from dunet import DUNet
from ibn_resnet import ibn_resnet50
from ibn_resnext import ibn_b_resnext50_32x4d
from resnext import resnext50_32x4d
from shufflenetv2p import shufflenetv2_plus
from ibn_shufflenetv2p import ibn_shufflenetv2_plus

try:
    import apex
except:
    print("apex is not installed successfully")
import time


def init_model(model_filename):
    device = torch.device("cuda")
    model = DeepLabV3Plus(nclass=[8,14], get_backbone=ibn_resnet50)
    
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
