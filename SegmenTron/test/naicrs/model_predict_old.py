import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import os
import math


class Scale():
    def __init__(self, crop_size, upsample_rate, stride):
        self.crop_size = crop_size
        self.upsample_rate = upsample_rate
        self.base_size = int(crop_size * upsample_rate)
        self.stride = stride


batch_size = 8
device = torch.device("cuda")
transform = transforms.Compose([
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def predict(model, input_path, output_dir, args):
    filename = os.path.basename(input_path)
    filename, _ = os.path.splitext(filename)
    image = Image.open(input_path).convert('RGB')
    image = torch.tensor(np.array(image), dtype=torch.uint8).permute(2, 0, 1).to(device)
    image = transform(image / 255.0)
    image = torch.unsqueeze(image, dim=0)
    
    with torch.no_grad():
        multi_scale_predict(model, image, filename, output_dir, args)
        torch.cuda.empty_cache()


def single_scale_predict_v1(scale: Scale, image, model, mode):
    width, height = image.size(2), image.size(3)
    image = F.interpolate(image, (int(width * scale.upsample_rate), int(height * scale.upsample_rate)),
                          mode='bilinear', align_corners=True)
    output = model(image, mode)[0]
    return output


def single_scale_predict_v2(scale: Scale, image, model, mode):
    origin_width, origin_height = image.size(2), image.size(3)
    
    pad_width = math.ceil((origin_width - scale.crop_size) / scale.stride) * scale.stride + scale.crop_size
    pad_height = math.ceil((origin_height - scale.crop_size) / scale.stride) * scale.stride + scale.crop_size
    image = F.interpolate(image, (pad_width, pad_height), mode='bilinear', align_corners=True)
    
    batch, channel, _, _ = image.shape
    images = F.unfold(image, kernel_size=scale.crop_size, stride=scale.stride)
    del image
    
    B, C_kw_kw, L = images.shape
    images = images.permute(0, 2, 1).contiguous()
    images = images.view(B, L, channel, scale.crop_size, scale.crop_size).squeeze(dim=0)
    
    batch_num = images.size(0) // batch_size + 1
    outputs = []
    for i in range(batch_num):
        batch = images[i * batch_size:(i + 1) * batch_size, ...]
        if batch.size(0) != 0:
            batch = F.interpolate(batch, (scale.base_size, scale.base_size), mode='bilinear', align_corners=True)
            output = model(batch, mode)[0]
            outputs.append(output)
    del images
    
    outputs = torch.cat(outputs, dim=0)
    nclass = outputs.size(1)
    outputs = outputs.unsqueeze(dim=0)
    outputs = outputs.view(B, L, nclass * scale.base_size * scale.base_size // 16).permute(0, 2, 1).contiguous()
    
    output = F.fold(outputs, output_size=(
        int(pad_width * scale.upsample_rate / 4), int(pad_height * scale.upsample_rate / 4)),
                    kernel_size=int(scale.base_size / 4), stride=int(scale.stride * scale.upsample_rate / 4))
    del outputs
    return output


def multi_scale_predict(model, image, filename, output_dir, args):
    scale1 = Scale(crop_size=256, upsample_rate=0.15, stride=256)
    # scale2 = Scale(crop_size=320, upsample_rate=1.0 * 0.8, stride=320 - args.stride)
    
    origin_width, origin_height = image.size(2), image.size(3)
    output = single_scale_predict_v1(scale1, image, model, args.mode)
    output = F.interpolate(output, (origin_width, origin_height), mode='bilinear', align_corners=True)
    
    predict = torch.argmax(output, dim=1).squeeze(0)
    
    if args.mode in ["01", "02", "03"]:
        predict = (predict + 1) * 100
        predict = predict.cpu().data.numpy()
        predict = predict.astype(np.uint16)
    else:
        predict += 1
        predict[predict >= 4] += 3
        predict = predict.to(torch.uint8).cpu().data.numpy()
        # predict = predict.astype(np.uint8)
    
    cv2.imwrite(os.path.join(output_dir, filename + ".png"), predict)