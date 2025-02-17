import torch
# a = torch.randn((2, 2, 6, 5))
b = torch.tensor([[1,2,3],[4,5,6]])
print(b.reshape(2,1,1,3).expand(-1, 4, 2, -1))

