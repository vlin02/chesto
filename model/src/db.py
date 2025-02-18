import io
import numpy as np


def tensor_to_binary(tensor):
    buffer = io.BytesIO()
    np.save(buffer, tensor)
    return buffer.getvalue()


def binary_to_tensor(binary_data):
    buffer = io.BytesIO(binary_data)
    return np.load(buffer, allow_pickle=True)
