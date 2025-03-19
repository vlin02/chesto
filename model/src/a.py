import torch

from net import NN
state_dict = torch.load('__tmp/0-1742265283.pt')
new_state_dict = {}
for k, v in state_dict.items():
    name = k.replace('_orig_mod.', '')  # Remove the prefix
    new_state_dict[name] = v

nn = NN()
nn.load_state_dict(state_dict2)