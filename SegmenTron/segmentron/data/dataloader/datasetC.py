"""NAICRS Semantic Segmentation Dataset."""
import os
import logging
import numpy as np
from PIL import Image
from .seg_data_base import SegmentationDataset
import torch

class NaicrsDatasetC(SegmentationDataset):
    
    BASE_DIR = 'datasetC'
    NUM_CLASS = 14

    def __init__(self, root='datasets/naicrs', split='test', mode=None, transform=None, **kwargs):
        super(NaicrsDatasetC, self).__init__(root, split, mode, transform, **kwargs)
        root = os.path.join(self.root, self.BASE_DIR)
        assert os.path.exists(root), "Please put the data in {SEG_ROOT}/datasets/naicrs"
        self.images, self.masks = _get_pairs(root, split)
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        logging.info('Found {} images in the folder {}'.format(len(self.images), root))
        logging.info(mode+": %d"%len(self.images))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index])
        mask = np.array(mask, dtype=np.int32)
        mask[mask==4] = 2 # 机场比例太低，合并进道路（即初赛的交通运输）
        mask[mask>=7] -= 3 # 火车站和光伏无数据，忽略
        mask = mask - 1  # 数据集中的标签是1-17，需要转换为0-16
        mask = Image.fromarray(mask)
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)

        # general resize, normalize and to Tensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask

    @property
    def classes(self):
        """Category names."""
        return ("水体", "道路", "建筑物", "停车场", "操场",
                "普通耕地", "农业大棚", "自然草地", "绿地绿化",
                "自然林", "人工林", "自然裸土", "人为裸土", "其它")

    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype('int32'))


def _get_pairs(folder, mode='train'):
    img_paths = []
    mask_paths = []
    if mode == 'train':
        f = open("datasets/naicrs/txt/trainC.txt", "r")
        train_filenames = f.readlines()
        train_filenames = [item.replace("\n", "") for item in train_filenames]
        f.close()
        img_folder = os.path.join(folder, 'trainval/images')
        mask_folder = os.path.join(folder, 'trainval/masks')
        for filename in train_filenames:
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
        return img_paths, mask_paths  # 取前90%作为训练集

    elif mode == 'val':
        f = open("datasets/naicrs/txt/valC.txt", "r")
        val_filenames = f.readlines()
        val_filenames = [item.replace("\n", "") for item in val_filenames]
        f.close()
        img_folder = os.path.join(folder, 'trainval/images')
        mask_folder = os.path.join(folder, 'trainval/masks')
        for filename in val_filenames:
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