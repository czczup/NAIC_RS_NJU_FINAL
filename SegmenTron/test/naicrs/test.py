from segmentron.models.backbones.ofa_v100_gpu64_6ms.main import ofa_v100_gpu64_6ms
from deeplabv3plus import DeepLabV3Plus
import torch
from PIL import Image
import numpy as np

# image_mean = np.array([0.485, 0.456, 0.406])
# image_std = np.array([0.229, 0.224, 0.225])
# image = Image.open("0.tif")
# image = image.resize((257, 257))
# image = np.array(image) / 255.0
# image = (image - image_mean) / image_std
# image = image.transpose((2, 0, 1))
# image = torch.Tensor(image).unsqueeze(0)

model = DeepLabV3Plus(nclass=[8, 14], get_backbone=ofa_v100_gpu64_6ms, channels=[32, 248])
state_dict = torch.load("../model/0046.pth", map_location=lambda storage, loc: storage)
msg = model.load_state_dict(state_dict, strict=False)
print(msg)
model.eval()
image = torch.ones(1, 3, 256, 256)
out = model(image, "03")[0][0]
print(out)
print(out[..., 37:43, 37:46])
print(out.shape)