import torch
from deeplabv3plus import DeepLabV3Plus
from segmentron.models.backbones.ofa_v100_gpu64_6ms.main import ofa_v100_gpu64_6ms, ibn_ofa_v100_gpu64_6ms
import argparse
from tools import fuse_module


def quantize_model(model_filename):
    model = DeepLabV3Plus(nclass=[8, 14], get_backbone=ofa_v100_gpu64_6ms, channels=[32, 248])
    model_path = "../model/%s" % model_filename
    print('load test model from {}'.format(model_path))
    msg = model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage), strict=False)
    model.eval()
    model = fuse_module(model)
    
    q_backend = 'fbgemm'
    qconfig = torch.quantization.get_default_qconfig(q_backend)
    torch.backends.quantized.engine = q_backend
    model.qconfig = qconfig
    
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)
    state_dict = {k: v for k, v in model.state_dict().items() if "scale" not in k and "zero_point" not in k}
    torch.save(state_dict, "../model/%s" % model_filename.split('.')[0]+"q.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segmentron')
    parser.add_argument('--model', type=str, help='model')
    args = parser.parse_args()
    quantize_model(args.model)
