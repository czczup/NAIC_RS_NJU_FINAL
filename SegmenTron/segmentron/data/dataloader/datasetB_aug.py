"""NAICRS Semantic Segmentation Dataset."""
import os
import logging
from PIL import Image
from .seg_data_base import SegmentationDataset
import torch
import numpy as np

class AugDatasetB(SegmentationDataset):

    BASE_DIR = 'datasetB'
    NUM_CLASS = 8

    def __init__(self, root='datasets/naicrs', split='test', mode=None, transform=None, **kwargs):
        super(AugDatasetB, self).__init__(root, split, mode, transform, **kwargs)
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
    
    
def _get_pairs(folder, mode='test'):
    img_paths = []
    mask_paths = []
    if mode == 'test':
        img_folder = os.path.join(folder, 'test/images')
        for filename in os.listdir(img_folder):
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".tif"):
                imgpath = os.path.join(img_folder, filename)
                img_paths.append(imgpath)
        return img_paths, mask_paths