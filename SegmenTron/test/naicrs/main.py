import importlib
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
from segmentron.utils.score import SegmentationMetric
import torch.nn.functional as F
import torch
import time
from tabulate import tabulate
import argparse

def generate_outputs(pyfile_path, input_paths, output_dir, args):
    if args.new:
        predict_py = pyfile_path+"model_predict"
    else:
        predict_py = pyfile_path+"model_predict_old"
    
    define_py = pyfile_path+"model_define"
    init_model = getattr(importlib.import_module(define_py), "init_model")
    predict = getattr(importlib.import_module(predict_py), "predict")

    model = init_model(args)
    start_time = time.time()
    for input_path in tqdm(input_paths):
        predict(model, input_path, output_dir, args)
    print("using time: %.2fs" %(time.time()-start_time))
    
    
def get_iou_datasetC():
    classes = ("水体", "交通", "建筑", "耕地",
               "草地", "林地", "裸土", "其它")
    device = torch.device("cuda")
    metric = SegmentationMetric(nclass=len(classes), distributed=False)
    metric.reset()
    mask_root = "../../../SegmenTron/datasets/naicrs/datasetC/trainval/masks"
    pred_root = "../../../SegmenTron/datasets/naicrs/datasetC/trainval/results"
    
    pred_files = os.listdir(pred_root)
    mask_files = [os.path.join(mask_root, filename) for filename in pred_files]
    pred_files = [os.path.join(pred_root, filename) for filename in pred_files]
    time_start = time.time()
    
    for i in range(len(pred_files)):
        pred = Image.open(pred_files[i])
        mask = Image.open(mask_files[i])
        pred = np.array(pred)
        pred = pred // 100 - 1
        mask = np.array(mask, dtype=np.int32)
        mask[mask == 4] = 2  # 机场比例太低，合并进道路（即初赛的交通运输）
        mask[mask >= 7] -= 3  # 火车站和光伏无数据，忽略
        mask = mask - 1
        mask[mask == 0] = 100  # 水体   -> 水体
        mask[mask == 1] = 200  # 道路   -> 道路
        mask[mask == 2] = 300  # 建筑物   -> 建筑物
        mask[mask == 3] = 800  # 停车场   -> 其它
        mask[mask == 4] = 800  # 操场    -> 其它
        mask[mask == 5] = 400  # 普通耕地 -> 耕地
        mask[mask == 6] = 400  # 农业大棚 -> 耕地
        mask[mask == 7] = 500  # 自然草地 -> 草地
        mask[mask == 8] = 500  # 绿地绿化 -> 草地
        mask[mask == 9] = 600  # 自然林   -> 林地
        mask[mask == 10] = 600  # 人工林   -> 林地
        mask[mask == 11] = 700  # 自然裸土 -> 裸土
        mask[mask == 12] = 700  # 人为裸土 -> 裸土
        mask[mask == 13] = 800  # 其它    -> 其它
        mask = mask // 100 - 1
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

def get_iou_datasetC_v2():
    classes = ("水体", "道路", "建筑物", "停车场", "操场",
               "普通耕地", "农业大棚", "自然草地", "绿地绿化",
               "自然林", "人工林", "自然裸土", "人为裸土", "其它")
    device = torch.device("cuda")
    metric = SegmentationMetric(nclass=len(classes), distributed=False)
    metric.reset()
    mask_root = "../../../SegmenTron/datasets/naicrs/datasetC/trainval/masks"
    pred_root = "../../../SegmenTron/datasets/naicrs/datasetC/trainval/results"
    pred_files = os.listdir(pred_root)
    mask_files = [os.path.join(mask_root, filename) for filename in pred_files]
    pred_files = [os.path.join(pred_root, filename) for filename in pred_files]
    time_start = time.time()
    
    for i in range(len(pred_files)):
        pred = Image.open(pred_files[i])
        mask = Image.open(mask_files[i])

        pred = np.array(pred)
        pred[pred >= 7] -= 3
        pred = pred - 1

        mask = np.array(mask)
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

def get_iou_datasetA_and_other(dataset):
    classes = ("水体", "交通", "建筑", "耕地",
               "草地", "林地", "裸土", "其它")
    device = torch.device("cuda")
    metric = SegmentationMetric(nclass=len(classes), distributed=False)
    metric.reset()
    if dataset == "datasetA":
        mask_root = "../../../SegmenTron/datasets/naicrs/datasetA/trainval/masks"
        pred_root = "../../../SegmenTron/datasets/naicrs/datasetA/trainval/results"
    else:
        mask_root = "../../../SegmenTron/datasets/"+dataset+"/masks"
        pred_root = "../../../SegmenTron/datasets/"+dataset+"/results"
        
    pred_files = os.listdir(pred_root)
    mask_files = [os.path.join(mask_root, filename) for filename in pred_files]
    pred_files = [os.path.join(pred_root, filename) for filename in pred_files]
    time_start = time.time()
    
    for i in range(len(pred_files)):
        pred = Image.open(pred_files[i])
        mask = Image.open(mask_files[i])
        pred = np.array(pred, dtype=np.int32)
        pred = pred // 100 - 1
        
        mask = np.array(mask, dtype=np.int32)
        mask = mask // 100 - 1
        
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segmentron')
    parser.add_argument('--dataset', type=str, help='dataset')
    parser.add_argument('--mode', type=str, help='mode')
    parser.add_argument('--model', type=str, help='model')
    parser.add_argument('--new', action="store_true", help='new')
    parser.add_argument('--stride', default=32, type=int, help='stride')
    parser.add_argument('--dali', action="store_true", help='dali')
    parser.add_argument('--quantize', action="store_true", help='quantize')
    args = parser.parse_args()
    
    if args.dataset == "datasetA":
        txt = "../../../SegmenTron/datasets/naicrs/txt/valA.txt"
        f = open(txt, "r")
        filenames = [item.replace("\n", "") for item in f.readlines()]
        data_dir = "../../../SegmenTron/datasets/naicrs/datasetA/trainval/images"
        output_dir = "../../../SegmenTron/datasets/naicrs/datasetA/trainval/results"
    elif args.dataset == "datasetC":
        txt = "../../../SegmenTron/datasets/naicrs/txt/valC.txt"
        f = open(txt, "r")
        filenames = [item.replace("\n", "") for item in f.readlines()]
        data_dir = "../../../SegmenTron/datasets/naicrs/datasetC/trainval/images"
        output_dir = "../../../SegmenTron/datasets/naicrs/datasetC/trainval/results"
    elif args.dataset == "deep-globe":
        data_dir = "../../../SegmenTron/datasets/" + args.dataset + "/images"
        output_dir = "../../../SegmenTron/datasets/" + args.dataset + "/results"
        filenames = [item for item in os.listdir(data_dir) if int(os.path.splitext(item)[0])>90000]
    else:
        data_dir = "../../../SegmenTron/datasets/"+args.dataset+"/images"
        output_dir = "../../../SegmenTron/datasets/"+args.dataset+"/results"
        filenames = os.listdir(data_dir)
        
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    input_paths = [os.path.join(data_dir, filename) for filename in filenames]
    generate_outputs('', input_paths=input_paths, output_dir=output_dir, args=args)
    
    if args.dataset == "datasetC":
        if args.mode in ["04", "05", "06"]:
            get_iou_datasetC_v2()
        else:
            get_iou_datasetC()
    else:
        get_iou_datasetA_and_other(args.dataset)