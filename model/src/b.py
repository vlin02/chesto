import torch

cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")