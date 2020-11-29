"""NAICRS Semantic Segmentation Dataset."""
import os
import logging
import numpy as np
from PIL import Image
from .seg_data_base import SegmentationDataset
import torch

class DatasetAC(SegmentationDataset):
    
    BASE_DIR_A = 'datasetA'
    BASE_DIR_C = 'datasetC'
    NUM_CLASS_A = 8
    NUM_CLASS_C = 14
    NUM_CLASS = 14
    
    def __init__(self, root='datasets/naicrs', split='test', mode=None, transform=None, **kwargs):
        super(DatasetAC, self).__init__(root, split, mode, transform, **kwargs)
        root_A = os.path.join(self.root, self.BASE_DIR_A)
        root_C = os.path.join(self.root, self.BASE_DIR_C)
        assert os.path.exists(root_A), "Please put the data in {SEG_ROOT}/datasets/naicrs"
        assert os.path.exists(root_C), "Please put the data in {SEG_ROOT}/datasets/naicrs"

        images_A, masks_A = _get_pairs(root_A, split + "A")
        images_C, masks_C = _get_pairs(root_C, split + "C")
        if len(images_A) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root_A + "\n")
        if len(images_C) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root_C + "\n")
        self.images = images_A + images_C
        self.masks = masks_A + masks_C
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        logging.info('Found {} images in the folder {}'.format(len(self.images), root))
        logging.info(mode+": %d"%len(self.images))

    def __getitem__(self, index):
        filename = self.images[index]
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index])
        mask = np.array(mask, dtype=np.int32)
        if "datasetA" in filename:  # datasetA
            mask = mask // 100 - 1  # 数据集中的标签是100-800，需要转换为0-7
        else:                       # datasetC
            mask[mask == 4] = 2     # 机场比例太低，合并进道路（即初赛的交通运输）
            mask[mask >= 7] -= 3    # 火车站和光伏无数据，忽略
            mask = mask - 1         # 数据集中的标签是1-17，需要转换为0-16 (0-13)
        mask = Image.fromarray(mask)
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)

        # general resize, normalize and to Tensor
        if self.transform is not None:
            img = self.transform(img)
            
        if "datasetA" in filename:
            mask_8 = mask
            mask_14 = np.zeros(mask_8.shape) - 1  # -1不监督
        else:
            mask_14 = mask
            mask_8 = mask.copy()
            mask_8[mask_14==0] = 0    # 水体   -> 水体
            mask_8[mask_14==1] = 1    # 道路   -> 道路
            mask_8[mask_14==2] = 2    # 建筑物  -> 建筑物
            mask_8[mask_14==3] = 7    # 停车场   -> 其它
            mask_8[mask_14==4] = 7    # 操场    -> 其它
            mask_8[mask_14==5] = 3    # 普通耕地 -> 耕地
            mask_8[mask_14==6] = 3    # 农业大棚 -> 耕地
            mask_8[mask_14==7] = 4    # 自然草地 -> 草地
            mask_8[mask_14==8] = 4    # 绿地绿化 -> 草地
            mask_8[mask_14==9] = 5    # 自然林   -> 林地
            mask_8[mask_14==10] = 5   # 人工林   -> 林地
            mask_8[mask_14==11] = 6   # 自然裸土 -> 裸土
            mask_8[mask_14==12] = 6   # 人为裸土 -> 裸土
            mask_8[mask_14==13] = 7   # 其它    -> 其它
        return img, torch.LongTensor(mask_8), torch.LongTensor(mask_14)

    @property
    def classes_C(self):
        """Category names."""
        return ("水体", "道路", "建筑物", "停车场", "操场",
                "普通耕地", "农业大棚", "自然草地", "绿地绿化",
                "自然林", "人工林", "自然裸土", "人为裸土", "其它")

    @property
    def classes_A(self):
        """Category names."""
        return ("水体", "交通运输", "建筑", "耕地",
                "草地", "林地", "裸土", "其它",)
    
def _get_pairs(folder, mode='train'):
    img_paths = []
    mask_paths = []
    f = open("datasets/naicrs/txt/%s.txt"%mode, "r")
    filenames = f.readlines()
    filenames = [item.replace("\n", "") for item in filenames]
    f.close()
    img_folder = os.path.join(folder, 'trainval/images')
    mask_folder = os.path.join(folder, 'trainval/masks')
    for filename in filenames:
        basename, _ = os.path.splitext(filename)
        if filename.endswith(".tif"):
            imgpath = os.path.join(img_folder, filename)
            maskname = basename + '.png'
            maskpath = os.path.join(mask_folder, maskname)
            if os.path.isfile(maskpath):
                img_paths.append(imgpath)
                mask_paths.append(maskpath)
            else:
                logging.info('cannot find the mask:', maskpath)
    return img_paths, mask_paths