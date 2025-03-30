import torch

a = torch.zeros((100, 2, 4))
b = torch.zeros((100, 6))

c = torch.cat([a.flatten(-2), b], dim=-1).mean(dim=0)
print(c)
