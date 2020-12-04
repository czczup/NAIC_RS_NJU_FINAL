import torch

state_dict = torch.load("model.pth")
print(len(state_dict))
for k, v in state_dict.items():
    print(k)
state_dict = {k:v for k,v in state_dict.items() if "aux" not in k}

state_dict_head = {k:v.half() for k,v in state_dict.items() if "head" in k}
torch.save(state_dict_head, "model_head_half.pth")
state_dict_encoder = {k:v.half() for k,v in state_dict.items() if "encoder" in k}
torch.save(state_dict_encoder, "model_encoder_half.pth")
state_dict_bn = {k:v.half() for k,v in state_dict.items() if "bn" not in k}
torch.save(state_dict_bn, "model_wo_bn.pth")