from model_define import DeepLabV3Plus, ibn_resnet50
import torch
from nni.compression.torch import FPGMPruner, ModelSpeedup
from tqdm import tqdm
import time


def purning():
    model_path = "../model/0001.pth"
    export_model_path = '../model/encoder.pth'
    export_mask_path = "../model/mask.pth"
    speed_mask_path = "../model/0001_speedup.pth"

    device = torch.device("cuda")
    dummy_input = torch.randn([1, 3, 256, 256]).to(device)

    model = DeepLabV3Plus(nclass=[8,14], get_backbone=ibn_resnet50).to(device)
    
    print('load test model from {}'.format(model_path))
    msg = model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage), strict=False)
    
    torch.cuda.synchronize()
    time_start = time.time()
    for i in tqdm(range(1000)):
        outs = model.encoder(dummy_input)
    torch.cuda.synchronize()
    print(time.time()-time_start)
    
    print('start pruning')
    configure_list = [{
        'sparsity': 0.5,
        'op_types': ['Conv2d']
    }]
    pruner = FPGMPruner(model, configure_list)
    model = pruner.compress()
    pruner.export_model(model_path=export_model_path, mask_path=export_mask_path)
    
    print('start speed up')
    m_speedup = ModelSpeedup(model.encoder, dummy_input, export_mask_path)
    m_speedup.speedup_model()

    torch.cuda.synchronize()
    time_start = time.time()
    for i in tqdm(range(1000)):
        outs = model.encoder(dummy_input)
    torch.cuda.synchronize()
    print(time.time() - time_start)
    
    torch.save(model.state_dict(), speed_mask_path)


if __name__ == '__main__':
    purning()
