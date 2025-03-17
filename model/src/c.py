import numpy as np
import ctypes
from numba import njit


def fast_bytes_to_float32_alt(byte_data):
    """
    Alternative implementation using unsafe direct pointer casting
    Requires contiguous C-ordered array
    """
    arr = np.empty(len(byte_data) // 4, dtype=np.float32)
    ctypes.memmove(
        arr.ctypes.data,
        ctypes.c_char_p(byte_data),
        len(byte_data)
    )
    return arr

# @njit
def process_bytes(byte_data):
    # Convert byte data to a NumPy array of unsigned 8-bit integers
    return np.frombuffer(byte_data, dtype=np.float32)

def read_binary_file(filename):
    with open(filename, 'rb') as file:
        binary_data = file.read()
    return binary_data

def fast_float32_frombuffer(buffer, count=None, offset=0):
  pass
    # """Optimized numpy.frombuffer for float32 with less overhead"""
    # if count is None:
    #     count = (len(buffer) - offset) // 4
    
    # if isinstance(buffer, bytes):
    #     # Direct pointer casting for bytes
    #     ptr = ctypes.cast(
    #         ctypes.c_char_p(buffer[offset:offset + count * 4]), 
    #         ctypes.POINTER(ctypes.c_float)
    #     )
    #     return np.ctypeslib.as_array(ptr, shape=(count,))
    # else:
    #     # For other buffer types
    #     return np.frombuffer(buffer, dtype=np.float32, count=count, offset=offset)

    
@profile
def main():
  x = [1, 0.6633333333333333, 0.6633333333333333, 0.23666666666666666, 0.20833333333333334, 0.34, 0.23666666666666666, 0.31166666666666665, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

  v = [x] * 10000
  for _ in range(10000):
    np.array(x)

  np.array(v)

  b = memoryview(read_binary_file("a"))
    
    # Pre-allocate a single bytearray
  concatenated = bytearray(1680000)
  
  # Copy each buffer into the correct position
  offset = 0
  a = b''.join([b] * 200 * 10)
  np.frombuffer(a)

  # print(np.frombuffer(bytes(concatenated), dtype=np.float32).shape)
     
main()