# import torch
# # Example usage
# lookup_table = torch.tensor([
#     [1, 2, 3],  # mapping for index 0
#     [4, 5, 6],  # mapping for index 1
#     [7, 8, 9]   # mapping for index 2
# ])  # shape: (3, 3)

# input_tensor = torch.tensor([
#     [0, 1],
#     [2, 0]
# ])  # shape: (2, 2)


# def lookup_table_expand(tensor_a, lookup_table):
#     """
#     Performs lookup operation where each value i in tensor_a is replaced by lookup_table[i]
    
#     Args:
#         tensor_a: Input tensor of any shape
#         lookup_table: Tensor of shape (M, N) where M is the maximum value in tensor_a + 1
        
#     Returns:
#         Tensor of shape (*tensor_a.shape, N)
#     """
#     return lookup_table[tensor_a]

# result = lookup_table_expand(input_tensor, lookup_table)
# print(result.shape)  # torch.Size([2, 2, 3])
# print(result)
# # tensor([[[1, 2, 3],
# #          [4, 5, 6]],
# #         [[7, 8, 9],
# #          [1, 2, 3]]])
import torch
# print(torch.tensor([-5]) * float('-inf'))
print(torch.tensor([[1,2],[3,4]]).repeat(2, 1))