import h5py
from sample import INPUT_KEYS
with h5py.File("sample3.hdf5", "r") as f:
  for i in range(100000):
    if f["move_option_mask"][i][0][0] != 1:
      print(f["move_option_mask"][i][0])