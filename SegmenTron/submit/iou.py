import os
from PIL import Image
import numpy as np
from segmentron.utils.score import SegmentationMetric
import torch.nn.functional as F
import torch
import time
from tabulate import tabulate


classes = ("水体", "道路", "建筑物", "停车场", "操场",
           "普通耕地", "农业大棚", "自然草地", "绿地绿化",
           "自然林", "人工林", "自然裸土", "人为裸土", "其它")

device = torch.device("cuda")
metric = SegmentationMetric(nclass=len(classes), distributed=False)
metric.reset()
mask_root = "../datasets/naicrs/datasetC/trainval/masks"
pred_root = "results"


pred_files = os.listdir(pred_root)
mask_files = [os.path.join(mask_root, filename) for filename in pred_files]
pred_files = [os.path.join(pred_root, filename) for filename in pred_files]
time_start = time.time()

for i in range(len(pred_files)):
    pred = Image.open(pred_files[i])
    mask = Image.open(mask_files[i])
    pred = np.array(pred, np.int32)
    pred[pred >= 7] -= 3
    pred = pred - 1
    
    mask = np.array(mask, np.int32)
    mask[mask == 4] = 2
    mask[mask >= 7] -= 3
    mask = mask - 1
    
    pred = torch.LongTensor(pred).unsqueeze(0).to(device)
    mask = torch.LongTensor(mask).unsqueeze(0).to(device)
    pred = F.one_hot(pred).permute(0, 3, 1, 2)
    metric.update(pred, mask)
    pixAcc, mIoU = metric.get()
    print("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
        i + 1, pixAcc * 100, mIoU * 100))
    
print('Eval use time: {:.3f} second'.format(time.time() - time_start))
pixAcc, mIoU, category_iou = metric.get(return_category_iou=True)
print('End validation pixAcc: {:.3f}, mIoU: {:.3f}'.format(
        pixAcc * 100, mIoU * 100))

headers = ['class id', 'class name', 'iou']
table = []
for i, cls_name in enumerate(classes):
    table.append([cls_name, category_iou[i]])
print('Category iou: \n {}'.format(tabulate(table, headers, tablefmt='grid', showindex="always",
                                                   numalign='center', stralign='center')))
try:
    weights = [0.02754302, 0.07850080, 0.12062946, 0.06670495,
               0.00267682, 0.22780647, 0.01259433, 0.12597127,
               0.02395271, 0.15588569, 0.07321313, 0.04918330,
               0.01127538, 0.02406267]
    
    weighted_iou = [weights[i]*category_iou[i] for i in range(len(weights))]
    fwiou = sum(weighted_iou)
    print("FWIoU: %.6f" % (fwiou*100))
except:
    pass