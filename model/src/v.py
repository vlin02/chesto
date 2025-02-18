import numpy as np

# Create sample data
x = np.arange(1000)  # Array of integers 0-999
y = np.random.rand(500)  # Array of 500 random floats

# Save arrays to npz file
np.savez('multiple_memmap.npz', x=x, y=y)

# Load as memmap
with np.load('multiple_memmap.npz', mmap_mode='r') as data:
    # Access the arrays
    x_memmap = data['x']  # Memory-mapped array
    y_memmap = data['y']  # Memory-mapped array
    
    # Now you can work with these memory-mapped arrays
    print(x_memmap[:10])  # First 10 elements of x
    print(y_memmap[:5])   # First 5 elements of y