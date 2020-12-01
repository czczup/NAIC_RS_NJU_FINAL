import torch

state_dict = torch.load("0021.pth")
print(len(state_dict))
for k, v in state_dict.items():
    print(k)
state_dict = {k:v.half() for k,v in state_dict.items() if "aux" not in k}
print(len(state_dict))
torch.save(state_dict, "0021_wo_aux.pth")