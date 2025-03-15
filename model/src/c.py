import torch

# Create a tensor S of shape (3, 4, 2)
S = torch.tensor([
    [[10, 11], [20, 21], [30, 31], [40, 41]],
    [[50, 51], [60, 61], [70, 71], [80, 81]],
    [[90, 91], [100, 101], [110, 111], [120, 121]]
])

# Create a tensor T of shape (3,) with indices
T = torch.tensor([1, 3, 0])

# Select S[i][T[i]] for each i
result = S[torch.arange(S.shape[0]), T]

print(S[[0,0],[0,1],[0,0]])

print(result)  # tensor([[20, 21], [80, 81], [90, 91]])