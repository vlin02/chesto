import torch
import io

# Create sample tensors
tensor1 = torch.tensor([1, 2, 3], dtype=torch.int32)
tensor2 = torch.tensor([4, 5, 6], dtype=torch.int32)
tensor3 = torch.tensor([7, 8, 9], dtype=torch.int32)

# Method 1: Using torch.save() and torch.load()
def concat_using_torch_save():
    # Save tensors to binary files
    torch.save(tensor1, 'tensor1.pt')
    torch.save(tensor2, 'tensor2.pt')
    torch.save(tensor3, 'tensor3.pt')
    
    # Load and concatenate
    loaded1 = torch.load('tensor1.pt')
    loaded2 = torch.load('tensor2.pt')
    loaded3 = torch.load('tensor3.pt')
    
    result = torch.cat([loaded1, loaded2, loaded3])
    print("Method 1 result:", result)
    return result

# Method 2: Using binary concatenation
def concat_binary_data():
    # Save to binary using BytesIO to avoid disk I/O
    buffer1 = io.BytesIO()
    buffer2 = io.BytesIO()
    buffer3 = io.BytesIO()
    
    torch.save(tensor1, buffer1)
    torch.save(tensor2, buffer2)
    torch.save(tensor3, buffer3)
    
    # Get binary data
    binary1 = buffer1.getvalue()
    binary2 = buffer2.getvalue()
    binary3 = buffer3.getvalue()
    
    # Store sizes for later loading
    sizes = [len(binary1), len(binary2), len(binary3)]
    
    # Concatenate binary data
    combined_binary = binary1 + binary2 + binary3
    
    # Load individual tensors from the combined binary
    result_tensors = []
    start = 0
    for size in sizes:
        buffer = io.BytesIO(combined_binary[start:start + size])
        tensor = torch.load(buffer)
        result_tensors.append(tensor)
        start += size
    
    result = torch.cat(result_tensors)
    print("Method 2 result:", result)
    return result

# Method 3: Using memory mapping (for large files)
def concat_using_memmap():
    import numpy as np
    
    # First save tensors as numpy arrays
    tensor1.numpy().tofile('tensor1.bin')
    tensor2.numpy().tofile('tensor2.bin')
    tensor3.numpy().tofile('tensor3.bin')
    
    # Create memory-mapped arrays
    mmap1 = np.memmap('tensor1.bin', dtype=np.int32, mode='r')
    mmap2 = np.memmap('tensor2.bin', dtype=np.int32, mode='r')
    mmap3 = np.memmap('tensor3.bin', dtype=np.int32, mode='r')
    
    # Convert back to torch tensors and concatenate
    result = torch.cat([
        torch.from_numpy(mmap1.copy()),
        torch.from_numpy(mmap2.copy()),
        torch.from_numpy(mmap3.copy())
    ])
    print("Method 3 result:", result)
    return result

# Add benchmarking
def benchmark_methods(iterations=5):
    import time
    
    methods = [
        ('Method 1 (torch.save/load)', concat_using_torch_save),
        ('Method 2 (binary concat)', concat_binary_data),
        ('Method 3 (memmap)', concat_using_memmap)
    ]
    
    results = {}
    
    for name, method in methods:
        times = []
        print(f"\nTesting {name}")
        
        for i in range(iterations):
            start_time = time.time()
            result = method()
            end_time = time.time()
            times.append(end_time - start_time)
            
            print(f"Iteration {i+1}: {times[-1]:.4f} seconds")
        
        avg_time = sum(times) / len(times)
        results[name] = {
            'avg_time': avg_time,
            'min_time': min(times),
            'max_time': max(times)
        }
    
    print("\nBenchmark Summary:")
    for name, stats in results.items():
        print(f"\n{name}:")
        print(f"  Average time: {stats['avg_time']:.4f} seconds")
        print(f"  Min time: {stats['min_time']:.4f} seconds")
        print(f"  Max time: {stats['max_time']:.4f} seconds")

if __name__ == "__main__":
    # Test all methods
    print("\nTesting individual methods:")
    result1 = concat_using_torch_save()
    result2 = concat_binary_data()
    result3 = concat_using_memmap()
    
    # Verify results match
    assert torch.equal(result1, result2)
    assert torch.equal(result2, result3)
    print("\nAll methods produced identical results!")
    
    # Run benchmarks
    print("\nRunning benchmarks...")
    benchmark_methods()