"""NAICRS Semantic Segmentation Dataset."""
import os
import logging
import numpy as np
from PIL import Image
from .seg_data_base import SegmentationDataset
import torch

class AugDatasetA(SegmentationDataset):
    
    BASE_DIR = 'datasetA'
    NUM_CLASS = 8

    def __init__(self, root='datasets/naicrs', split='test', mode=None, transform=None, **kwargs):
        super(AugDatasetA, self).__init__(root, split, mode, transform, **kwargs)
        root = os.path.join(self.root, self.BASE_DIR)
        assert os.path.exists(root), "Please put the data in {SEG_ROOT}/datasets/naicrs"
        self.images, self.masks = _get_pairs(root, split)
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        logging.info('Found {} images in the folder {}'.format(len(self.images), root))
        logging.info(mode+": %d"%len(self.images))

    def __getitem__(self, index):
        filename = os.path.basename(self.images[index])
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, filename

        elif self.mode == 'aug_test':
            img0 = self.transform(self._val_flip(img, index=0))
            img1 = self.transform(self._val_flip(img, index=1))
            img2 = self.transform(self._val_flip(img, index=2))
            img3 = self.transform(self._val_flip(img, index=3))
            return img0, img1, img2, img3, filename

        mask = Image.open(self.masks[index])
        mask = np.array(mask, dtype=np.int32) // 100 - 1  # 数据集中的标签是100-800，需要转换为0-7
        mask = Image.fromarray(mask)

        # synchrosized transform
        if self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
            if self.transform is not None:
                img = self.transform(img)
            return img, mask

        elif self.mode == 'aug_val':
            img0 = self.transform(self._val_flip(img, index=0))
            img1 = self.transform(self._val_flip(img, index=1))
            img2 = self.transform(self._val_flip(img, index=2))
            img3 = self.transform(self._val_flip(img, index=3))
            mask = self._mask_transform(mask)
            return img0, img1, img2, img3, mask, filename

    def _val_flip(self, img, index):
        if index == 0:
            pass
        elif index == 1: # flip left right
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif index == 2: # flip top bottom
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        elif index == 3: # flip top bottom and left right
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        return img

    @property
    def classes(self):
        """Category names."""
        return ("水体", "交通运输", "建筑", "耕地",
                "草地", "林地", "裸土", "其它",)

    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype('int32'))

def _get_pairs(folder, split='train'):
    img_paths = []
    mask_paths = []
    if split == 'val' or split == 'train':
        f = open("datasets/naicrs/txt/%sA.txt"%split, "r")
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
        return img_paths, mask_paths  # 取前10%作为验证集

    elif split == 'test':
        img_folder = os.path.join(folder, 'test/images')
        for filename in os.listdir(img_folder):
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".tif"):
                imgpath = os.path.join(img_folder, filename)
                img_paths.append(imgpath)
        return img_paths, mask_paths