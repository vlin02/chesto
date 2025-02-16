import torch
import numpy as np
# Convert numpy to tensor and save
np_array = np.array([1, 2, 3])
torch.save(torch.from_numpy(np_array), 'data.pt')

# To load back as numpy:
loaded_tensor = torch.load('data.pt')
loaded_array = loaded_tensor.numpy()

mask = torch.randint(0, 5, (2, 6, 4), device='cuda')
print(mask.clamp(max=1))

print(torch.arange(2).repeat(4))