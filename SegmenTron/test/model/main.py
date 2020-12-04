import torch

state_dict = torch.load("0028.pth")
print(len(state_dict))
for k, v in state_dict.items():
    print(k)
state_dict_head = {k:v.half() for k,v in state_dict.items() if "head" in k}
torch.save(state_dict_head, "0028_head_half.pth")
state_dict_encoder = {k:v.half() for k,v in state_dict.items() if "encoder" in k}
torch.save(state_dict_encoder, "0028_encoder_half.pth")
state_dict_bn = {k:v.half() for k,v in state_dict.items() if "bn" not in k}
torch.save(state_dict_bn, "0028_wo_bn.pth")