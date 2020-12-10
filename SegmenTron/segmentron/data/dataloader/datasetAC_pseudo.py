"""NAICRS Semantic Segmentation Dataset."""
import logging
from .seg_data_base import SegmentationDataset
import os
import random
import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter
from ...config import cfg

class PseudoDatasetAC(SegmentationDataset):
    
    BASE_DIR_A = 'datasetA'
    BASE_DIR_C = 'datasetC'
    NUM_CLASS_A = 8
    NUM_CLASS_C = 14
    NUM_CLASS = 14
    
    def __init__(self, root='datasets/naicrs', split='test', mode=None, transform=None, **kwargs):
        super(PseudoDatasetAC, self).__init__(root, split, mode, transform, **kwargs)
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
        pseudo = None
        if "datasetA" in filename:
            mask = mask // 100 - 1
        else:
            mask[mask == 4] = 2
            mask[mask >= 7] -= 3
            mask = mask - 1
        mask = Image.fromarray(mask)
        # synchrosized transform
        if self.mode == 'train': # 训练时加入14类的伪标签
            if "datasetA" in filename:
                pseudo = Image.open(self.masks[index].replace("masks", "masks_0039"))
                pseudo = np.array(pseudo, dtype=np.int32)
                pseudo = Image.fromarray(pseudo)
                img, mask, pseudo = self._sync_transform(img, mask, pseudo)
            else:
                img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)

        # general resize, normalize and to Tensor
        if self.transform is not None:
            img = self.transform(img)
            
        if "datasetA" in filename:
            mask_8 = mask
            mask_14 = np.zeros(mask_8.shape) - 1 if pseudo is None else pseudo
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

    def _sync_transform(self, img, mask, pseudo=None):
        # random mirror
        if cfg.AUG.MIRROR and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if pseudo is not None:
                pseudo = pseudo.transpose(Image.FLIP_LEFT_RIGHT)

        if cfg.AUG.MIRROR and random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            if pseudo is not None:
                pseudo = pseudo.transpose(Image.FLIP_TOP_BOTTOM)
                
        crop_size = self.crop_size
    
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * cfg.AUG.RANDOM_SCALE[0]),
                                    int(self.base_size * cfg.AUG.RANDOM_SCALE[1]))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        if pseudo is not None:
            pseudo = pseudo.resize((ow, oh), Image.NEAREST)

        # pad crop
        if short_size < min(crop_size):
            padh = crop_size[0] - oh if oh < crop_size[0] else 0
            padw = crop_size[1] - ow if ow < crop_size[1] else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=-1)
            if pseudo is not None:
                pseudo = ImageOps.expand(pseudo, border=(0, 0, padw, padh), fill=-1)

        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size[1])
        y1 = random.randint(0, h - crop_size[0])
        img = img.crop((x1, y1, x1 + crop_size[1], y1 + crop_size[0]))
        mask = mask.crop((x1, y1, x1 + crop_size[1], y1 + crop_size[0]))
        if pseudo is not None:
            pseudo = pseudo.crop((x1, y1, x1 + crop_size[1], y1 + crop_size[0]))

        # gaussian blur as in PSP
        if cfg.AUG.BLUR_PROB > 0 and random.random() < cfg.AUG.BLUR_PROB:
            radius = cfg.AUG.BLUR_RADIUS if cfg.AUG.BLUR_RADIUS > 0 else random.random()
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    
        # color jitter
        if self.color_jitter and random.random() < cfg.AUG.COLOR_JITTER_PROB:
            img = self.color_jitter(img)
    
        mask = mask.resize((self.supervise_size, self.supervise_size), Image.NEAREST)
        if pseudo is not None:
            pseudo = pseudo.resize((self.supervise_size, self.supervise_size), Image.NEAREST)

        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        if pseudo is not None:
            pseudo = self._mask_transform(pseudo)
            return img, mask, pseudo
        else:
            return img, mask
    
    
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