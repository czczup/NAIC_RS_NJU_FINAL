import os
import torch
import logging
import torch.utils.model_zoo as model_zoo

from ...utils.download import download
from ...utils.registry import Registry
from ...config import cfg

BACKBONE_REGISTRY = Registry("BACKBONE")
BACKBONE_REGISTRY.__doc__ = """
Registry for backbone, i.e. resnet.

The registered object will be called with `obj()`
and expected to return a `nn.Module` object.
"""

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnet50c': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/resnet50-25c4b509.pth',
    'resnet101c': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/resnet101-2a57e44d.pth',
    'resnet152c': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/resnet152-0d43d698.pth',
    'xception65': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/tf-xception65-270e81cf.pth',
    'hrnet_w18_small_v1': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/hrnet-w18-small-v1-08f8ae64.pth',
    'mobilenet_v2': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/mobilenetV2-15498621.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'resnest50':'https://s3.us-west-1.wasabisys.com/resnest/torch/resnest50-528c19ca.pth',
    'resnest101': 'https://s3.us-west-1.wasabisys.com/resnest/torch/resnest101-22405ba7.pth',
}


def load_backbone_pretrained(model, backbone):
    if cfg.PHASE == 'train' and cfg.TRAIN.BACKBONE_PRETRAINED and (not cfg.TRAIN.PRETRAINED_MODEL_PATH):
        if os.path.isfile(cfg.TRAIN.BACKBONE_PRETRAINED_PATH):
            logging.info('Load backbone pretrained model from {}'.format(
                cfg.TRAIN.BACKBONE_PRETRAINED_PATH
            ))
            msg = model.load_state_dict(torch.load(cfg.TRAIN.BACKBONE_PRETRAINED_PATH), strict=False)
            logging.info(msg)
        elif backbone not in model_urls:
            logging.info('{} has no pretrained model'.format(backbone))
            return
        else:
            logging.info('load backbone pretrained model from url..')
            try:
                msg = model.load_state_dict(model_zoo.load_url(model_urls[backbone]), strict=False)
            except Exception as e:
                logging.warning(e)
                logging.info('Use torch download failed, try custom method!')

                msg = model.load_state_dict(torch.load(download(model_urls[backbone],
                        path=os.path.join(torch.hub._get_torch_home(), 'checkpoints'))), strict=False)
            logging.info(msg)


def get_segmentation_backbone(backbone, norm_layer=torch.nn.BatchNorm2d, pretrain=True):
    """
    Built the backbone model, defined by `cfg.MODEL.BACKBONE`.
    """
    model = BACKBONE_REGISTRY.get(backbone)(norm_layer)
    if pretrain:
        logging.info('Load backbone pretrained model from {}'.format(
            cfg.TRAIN.BACKBONE_PRETRAINED_PATH
        ))
        load_checkpoint(model, cfg.TRAIN.BACKBONE_PRETRAINED_PATH, strict=False, logger=logging)
    return model


def load_checkpoint(model, filename, map_location=None, strict=False, logger=None):
    checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    # strip prefix of state_dict
    if "shufflenet" in filename.lower():
        translater = open("pretrain/shufflenetv2p.csv", "r")
        content = [item[:-1].split(",") for item in translater.readlines()]
        translater.close()
        old2new = {item[1]:item[0] for item in content}
        state_dict = {old2new[k]:v for k, v in state_dict.items() if k in old2new}
    elif "vovnet" in filename.lower():
        state_dict = {k.replace("bottom_up.", "").replace("backbone.", ""): v for k, v in state_dict.items()}
    else:
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
        if list(state_dict.keys())[0].startswith('backbone.'):
            state_dict = {k[9:]: v for k, v in checkpoint['state_dict'].items()}
    # load state_dict
    from mmcv.runner import load_state_dict
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint
