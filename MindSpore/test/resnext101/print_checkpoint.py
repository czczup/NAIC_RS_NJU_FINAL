import os
import sys
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(os.path.split(cur_path)[0])[0]
sys.path.append(root_path)

from mindspore import context
from mindspore.train.serialization import save_checkpoint, load_checkpoint
from src.nets.backbones.resnext import resnext50_32x4d as ms_resnext50
from src.nets.backbones.resnext import resnext101_32x8d as ms_resnext101
from segmentron.models.backbones.resnext import resnext50_32x4d as pt_resnext50
from segmentron.models.backbones.resnext import resnext101_32x8d as pt_resnext101

def key_mapping(key):
    key = key.replace("bn0.weight", "bn0.gamma")
    key = key.replace("bn1.weight", "bn1.gamma")
    key = key.replace("bn2.weight", "bn2.gamma")
    key = key.replace("bn3.weight", "bn3.gamma")

    key = key.replace("bn0.bias", "bn0.beta")
    key = key.replace("bn1.bias", "bn1.beta")
    key = key.replace("bn2.bias", "bn2.beta")
    key = key.replace("bn3.bias", "bn3.beta")
    
    key = key.replace("running_mean", "moving_mean")
    key = key.replace("running_var", "moving_variance")
    
    key = key.replace("downsample.1.weight", "downsample.1.gamma")
    key = key.replace("downsample.1.bias", "downsample.1.beta")
    
    key = key.replace("aspp0.conv", "aspp0.0")
    key = key.replace("aspp0.bn.weight", "aspp0.1.gamma")
    key = key.replace("aspp0.bn.bias", "aspp0.1.beta")
    key = key.replace("aspp0.bn", "aspp0.1")

    key = key.replace("bn_depth.weight", "bn_depth.gamma")
    key = key.replace("bn_depth.bias", "bn_depth.beta")
    key = key.replace("bn_point.weight", "bn_point.gamma")
    key = key.replace("bn_point.bias", "bn_point.beta")
    
    key = key.replace("image_pooling.conv", "image_pooling.0")
    
    key = key.replace("image_pooling.bn.weight", "image_pooling.1.gamma")
    key = key.replace("image_pooling.bn.bias", "image_pooling.1.beta")
    key = key.replace("image_pooling.bn", "image_pooling.1")

    key = key.replace("bn.weight", "bn.gamma")
    key = key.replace("bn.bias", "bn.beta")

    key = key.replace("auxlayer.block.1.weight", "auxlayer.block.1.gamma")
    key = key.replace("auxlayer.block.1.bias", "auxlayer.block.1.beta")

    key = key.replace("auxlayer2.block.1.weight", "auxlayer2.block.1.gamma")
    key = key.replace("auxlayer2.block.1.bias", "auxlayer2.block.1.beta")

    return key

    
    
if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False, device_id=0)
    ms_model = ms_resnext101()
    save_checkpoint(ms_model, "resnext101.ckpt")
    param_dict = load_checkpoint("resnext101.ckpt")
    ms_keys = list(param_dict.keys())
    # for k, v in param_dict.items():
    #     print(k, v.shape)
        
    # print(len(param_dict.keys()))
    
    pt_model = pt_resnext101()
    state_dict = pt_model.state_dict()
    state_dict = {k:v for k, v in state_dict.items() if "num_batches_tracked" not in k}
    pt_keys = list(state_dict.keys())
    # for k, v in state_dict.items():
    #     print(k, v.shape)
    
    for i in range(len(ms_keys)):
        ms_key = ms_keys[i]
        pt_key = key_mapping(pt_keys[i])
        print(ms_key, pt_key)

    # print(len(state_dict.keys()))
