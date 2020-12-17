import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import os
import math
from multiprocessing import Process, Queue

a0 = 0
b0 = 0
result_queue = Queue()
result_path_queue = Queue()

input_path_queue = Queue()

image_queue = Queue()
image_path_queue = Queue()
flag = True

class Scale():
    def __init__(self, crop_size, upsample_rate, stride):
        self.crop_size = crop_size
        self.upsample_rate = upsample_rate
        self.base_size = int(crop_size * upsample_rate)
        self.stride = stride


# TODO:
threshold_a = 9000 # being 290000
threshold_b = 9000 # being 20000

batch_size = 16
device = torch.device("cuda")
transform = transforms.Compose([
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def writer():
    a1 = 0
    while a1 < threshold_a:
        # print("writer:", a1)
        result = result_queue.get()
        w, h = result.shape
        filename = result_path_queue.get()
        cv2.imwrite(filename, result)
        a1 += (w * h) / (256 * 256)

p_writer = Process(target=writer, args=())
p_writer.start()

def reader():
    b1 = 0
    while b1 < threshold_b:
        # print("reader:", b1)
        input_path, result_path = input_path_queue.get()
        image = Image.open(input_path).convert('RGB')
        image_queue.put(image)
        image_path_queue.put(result_path)
        b1 += 1

p_reader = Process(target=reader, args=())
p_reader.start()

def predict(model, input_path, output_dir, args):
    filename = os.path.basename(input_path)
    filename, _ = os.path.splitext(filename)
    result_path = os.path.join(output_dir, filename + ".png")
    global b0, flag
    # print("predict:", b0)
    if b0 < threshold_b:
        input_path_queue.put((input_path, result_path))
        b0 += 1
    else:
        if flag:
            for i in range(threshold_b):
                if i % 100 == 0:
                    print(i, threshold_b)
                image = image_queue.get()
                # print("predict: i =", i)
                result_path = image_path_queue.get()
                inference(model, image, result_path, args)
            flag = False
            
        image = Image.open(input_path).convert('RGB')
        inference(model, image, result_path, args)
    


def inference(model, image, path, args):
    image = torch.tensor(np.array(image), dtype=torch.uint8).permute(2, 0, 1).to(device)
    image = transform(image / 255.0)
    image = torch.unsqueeze(image, dim=0)
    global a0
    with torch.no_grad():
        predict = multi_scale_predict(model, image, args)
        if a0 < threshold_a:
            result_path_queue.put(path)
            result_queue.put(predict)
            w, h = predict.shape
            a0 += (w * h) / (256 * 256)
        else:
            p_writer.join()
            cv2.imwrite(path, predict)
        torch.cuda.empty_cache()

def single_scale_predict_v1(scale: Scale, image, model):
    width, height = image.size(2), image.size(3)
    image = F.interpolate(image, (int(width * scale.upsample_rate), int(height * scale.upsample_rate)),
                          mode='bilinear', align_corners=True)
    output = model(image)[0]
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


def multi_scale_predict(model, image, args):
    scale1 = Scale(crop_size=256, upsample_rate=1.0, stride=256 - args.stride)
    # scale2 = Scale(crop_size=320, upsample_rate=1.0 * 0.8, stride=320 - args.stride)
    
    origin_width, origin_height = image.size(2), image.size(3)
    output1 = single_scale_predict_v2(scale1, image, model, mode=args.mode)
    # output2 = single_scale_predict_v2(scale2, image, model, mode=args.mode)
    # output2 = F.interpolate(output2, (output1.size(2), output1.size(3)), mode='bilinear', align_corners=True)
    # output = output1 + output2
    output = output1
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
    return predict
