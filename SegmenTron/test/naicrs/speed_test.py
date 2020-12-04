import torch
from deeplabv3plus import DeepLabV3Plus
from dunet import DUNet
from ibn_resnet import ibn_resnet50
from ibn_resnext import ibn_b_resnext50_32x4d
from resnext import resnext50_32x4d
from torchvision import transforms
from segmentron.models.backbones.ofa_1080ti_gpu64_27ms.main import ofa_1080ti_gpu64_27ms
from segmentron.models.backbones.tiny_ofa_1080ti_gpu64_27ms.main import tiny_ofa_1080ti_gpu64_27ms, ibn_tiny_ofa_1080ti_gpu64_27ms
from segmentron.models.backbones.shufflenetv2p import shufflenetv2_plus
from segmentron.models.backbones.ibn_shufflenetv2p import ibn_shufflenetv2_plus
from segmentron.models.backbones.ofa_note10_lat_8ms.main import ofa_note10_lat_8ms, ibn_ofa_note10_lat_8ms
from segmentron.models.backbones.ofa_v100_gpu64_6ms.main import ofa_v100_gpu64_6ms, ibn_ofa_v100_gpu64_6ms
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import time
try:
    import apex
except:
    print("apex is not installed")


def fuse_conv_bn(conv, bn):
    """During inference, the functionary of batch norm layers is turned off but
    only the mean and var alone channels are used, which exposes the chance to
    fuse it with the preceding conv layers to save computations and simplify
    network structures."""
    conv_w = conv.weight
    conv_b = conv.bias if conv.bias is not None else torch.zeros_like(
        bn.running_mean)

    factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    conv.weight = nn.Parameter(conv_w *
                               factor.reshape([conv.out_channels, 1, 1, 1]))
    conv.bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)
    return conv


def fuse_module(m):
    last_conv = None
    last_conv_name = None

    for name, child in m.named_children():
        if isinstance(child, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            fused_conv = fuse_conv_bn(last_conv, child)
            m._modules[last_conv_name] = fused_conv
            # To reduce changes, set BN as Identity instead of deleting it.
            m._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, nn.Conv2d):
            last_conv = child
            last_conv_name = name
        else:
            fuse_module(child)
    return m


def test_model(model):
    device = torch.device("cuda")
    # try:
    #     model = apex.amp.initialize(model.cuda(), opt_level="O3")
    # except:
    #     pass
    model.eval()
    model = model.to(device)

    image = Image.open("0.tif").convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    image = transform(image)
    image = torch.unsqueeze(image, dim=0)
    batch_size = 1
    images = [image] * batch_size
    images = torch.cat(images, 0).to(device)
    
    for i in range(500):
        outs = model(images, mode="03")
        
    torch.cuda.synchronize()
    time_start = time.time()
    for i in range(1000):
        outs = model(images, mode="03")
    torch.cuda.synchronize()
    return (time.time() - time_start) / batch_size


if __name__ == '__main__':
    results = []
    num = 1000
    # model = DeepLabV3Plus(nclass=[8,14], get_backbone=ibn_resnet50, channels=[256, 2048])
    # model = fuse_module(model)
    # results.append("ibn_resnet50: %.2fs/%dpics" % (test_model(model), num))
    # print(results[-1])
    #
    # model = DeepLabV3Plus(nclass=[8, 14], get_backbone=ibn_b_resnext50_32x4d, channels=[256, 2048])
    # model = fuse_module(model)
    # results.append("ibn_b_resnext50_32x4d: %.2fs/%dpics" % (test_model(model), num))
    # print(results[-1])
    #
    # model = DeepLabV3Plus(nclass=[8, 14], get_backbone=shufflenetv2_plus, channels=[68, 1280])
    # model = fuse_module(model)
    # results.append("shufflenetv2_plus: %.2fs/%dpics" % (test_model(model), num))
    # print(results[-1])
    #
    # model = DeepLabV3Plus(nclass=[8, 14], get_backbone=ibn_shufflenetv2_plus, channels=[68, 1280])
    # model = fuse_module(model)
    # results.append("ibn_shufflenetv2_plus: %.2fs/%dpics" % (test_model(model), num))
    # print(results[-1])
    #
    # model = DeepLabV3Plus(nclass=[8, 14], get_backbone=ofa_1080ti_gpu64_27ms, channels=[32, 1664])
    # model = fuse_module(model)
    # results.append("ofa_1080ti_gpu64_27ms: %.2fs/%dpics" % (test_model(model), num))
    # print(results[-1])
    #
    # model = DeepLabV3Plus(nclass=[8,14], get_backbone=tiny_ofa_1080ti_gpu64_27ms, channels=[32, 416])
    # model = fuse_module(model)
    # results.append("tiny_ofa_1080ti_gpu64_27ms: %.2fs/%dpics" % (test_model(model), num))
    # print(results[-1])
    #
    # model = DeepLabV3Plus(nclass=[8,14], get_backbone=ibn_tiny_ofa_1080ti_gpu64_27ms, channels=[32, 416])
    # model = fuse_module(model)
    # results.append("ibn_tiny_ofa_1080ti_gpu64_27ms: %.2fs/%dpics" % (test_model(model), num))
    # print(results[-1])
    #
    model = DeepLabV3Plus(nclass=[8,14], get_backbone=ofa_note10_lat_8ms, channels=[24, 160])
    model = fuse_module(model)
    results.append("ofa_note10_lat_8ms: %.2fs/%dpics" % (test_model(model), num))
    print(results[-1])

    model = DeepLabV3Plus(nclass=[8,14], get_backbone=ibn_ofa_note10_lat_8ms, channels=[24, 160])
    model = fuse_module(model)
    results.append("ibn_ofa_note10_lat_8ms: %.2fs/%dpics" % (test_model(model), num))
    print(results[-1])

    model = DeepLabV3Plus(nclass=[8,14], get_backbone=ofa_v100_gpu64_6ms, channels=[32, 248])
    # model = fuse_module(model)
    results.append("ofa_v100_gpu64_6ms: %.2fs/%dpics" % (test_model(model), num))
    print(results[-1])

    model = DeepLabV3Plus(nclass=[8,14], get_backbone=ibn_ofa_v100_gpu64_6ms, channels=[32, 248])
    torch.save({k: v.half() for k, v in model.state_dict().items()}, "model.pth")
    # model = fuse_module(model)
    results.append("ibn_ofa_v100_gpu64_6ms: %.2fs/%dpics" % (test_model(model), num))
    print(results[-1])

    print("*********")
    for result in results:
        print(result)
