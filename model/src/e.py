import numpy as np

# Example dictionary of tensors
tensor_dict = {
    'tensor_a': np.array([[1, 2], [3, 4]]),
    'tensor_b': np.array([5, 6, 7]),
    'tensor_c': np.array([8, 9])
}

# Save the dictionary of tensors to a .npz file
np.savez('tensor_dict.npz', **tensor_dict)