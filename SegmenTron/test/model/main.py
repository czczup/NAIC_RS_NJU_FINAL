import torch

state_dict = torch.load("resnet50.pth", map_location=torch.device('cpu'))

# print(len(state_dict))
# for k, v in state_dict.items():
#     if "scale" not in k and "zero_point" not in k:
#         print(k, v.dtype)
# new_state_dict = dict()
# for k, v in state_dict.items():
#     if "weight":
#         new_state_dict[k] = v.dequantize()
#     else:
#         new_state_dict[k] = v
# state_dict = new_state_dict

state_dict_head = {k:v for k,v in state_dict.items() if "aux" not in k}
torch.save(state_dict_head, "reesnet50_without_aux.pth")
# state_dict_encoder = {k:v for k,v in state_dict.items() if "encoder" in k}
# torch.save(state_dict_encoder, "model_encoder_half.pth")
# state_dict_bn = {k:v for k,v in state_dict.items() if "bn" not in k}
# torch.save(state_dict_bn, "model_wo_bn.pth")