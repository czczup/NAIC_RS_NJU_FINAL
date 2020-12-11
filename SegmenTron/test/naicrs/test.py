from segmentron.models.backbones.ofa_v100_gpu64_6ms.main import ofa_v100_gpu64_6ms
from deeplabv3plus import DeepLabV3Plus
import torch


model = DeepLabV3Plus(nclass=[8, 14], get_backbone=ofa_v100_gpu64_6ms, channels=[32, 248])
state_dict = torch.load("../model/0046.pth", map_location=lambda storage, loc: storage)
msg = model.load_state_dict(state_dict, strict=False)
print(msg)
model.eval()
input = torch.ones(1, 3, 640, 640)
out = model(input, "04")