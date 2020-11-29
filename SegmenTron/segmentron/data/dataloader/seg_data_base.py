"""Base segmentation dataset"""
import os
import random
import numpy as np
import torchvision
import torch
from PIL import Image, ImageOps, ImageFilter
from ...config import cfg

__all__ = ['SegmentationDataset']


class SegmentationDataset(object):
    """Segmentation Base Dataset"""

    def __init__(self, root, split, mode, transform, base_size=520, crop_size=480, extra_data_probability=0):
        super(SegmentationDataset, self).__init__()
        self.root = os.path.join(cfg.ROOT_PATH, root)
        self.transform = transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = self.to_tuple(crop_size)
        self.color_jitter = self._get_color_jitter()
        self.supervise_size = int(cfg.TRAIN.SUPERVISE_SIZE)

    def to_tuple(self, size):
        if isinstance(size, (list, tuple)):
            return tuple(size)
        elif isinstance(size, (int, float)):
            return tuple((size, size))
        else:
            raise ValueError('Unsupport datatype: {}'.format(type(size)))

    def _get_color_jitter(self):
        color_jitter = cfg.AUG.COLOR_JITTER
        if color_jitter is None:
            return None
        if isinstance(color_jitter, (list, tuple)):
            # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
            # or 4 if also augmenting hue
            assert len(color_jitter) in (3, 4)
        else:
            # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
            color_jitter = (float(color_jitter),) * 3
        return torchvision.transforms.ColorJitter(*color_jitter)

    def _val_sync_transform(self, img, mask):
        img = img.resize((self.base_size, self.base_size), Image.BILINEAR)
        mask = mask.resize((self.supervise_size, self.supervise_size), Image.NEAREST)
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror
        if cfg.AUG.MIRROR and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if cfg.AUG.MIRROR and random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
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

        # pad crop
        if short_size < min(crop_size):
            padh = crop_size[0] - oh if oh < crop_size[0] else 0
            padw = crop_size[1] - ow if ow < crop_size[1] else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=-1)

        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size[1])
        y1 = random.randint(0, h - crop_size[0])
        img = img.crop((x1, y1, x1 + crop_size[1], y1 + crop_size[0]))
        mask = mask.crop((x1, y1, x1 + crop_size[1], y1 + crop_size[0]))

        # gaussian blur as in PSP
        if cfg.AUG.BLUR_PROB > 0 and random.random() < cfg.AUG.BLUR_PROB:
            radius = cfg.AUG.BLUR_RADIUS if cfg.AUG.BLUR_RADIUS > 0 else random.random()
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        # color jitter
        if self.color_jitter and random.random() < cfg.AUG.COLOR_JITTER_PROB:
            img = self.color_jitter(img)
            
        # random rotate
        if cfg.AUG.ROTATE:
            img, mask = random_rotate(img, mask)

        mask = mask.resize((self.supervise_size, self.supervise_size), Image.NEAREST)
        
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        return np.array(mask).astype('int32')

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0
    
    def __len__(self):
        return len(self.images)


def random_rotate(img, mask):
    rotate_angle = random.choice([0, 90, 180, 270])
    img = img.rotate(rotate_angle)
    mask = mask.rotate(rotate_angle)
    return img, mask