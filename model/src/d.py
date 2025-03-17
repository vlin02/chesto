import numpy as np
import time

list_size = 1000000
my_list = list(range(list_size))

# Using numpy.array()
start_time = time.time()
array_np_array = np.array(my_list)
end_time = time.time()
print(f"numpy.array() time: {end_time - start_time} seconds")

# Using numpy.asarray()
start_time = time.time()
array_np_asarray = np.asarray(my_list)
end_time = time.time()
print(f"numpy.asarray() time: {end_time - start_time} seconds")

# Using numpy.fromiter()
start_time = time.time()
array_np_fromiter = np.fromiter(my_list, dtype=int)
end_time = time.time()
print(f"numpy.fromiter() time: {end_time - start_time} seconds")