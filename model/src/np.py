import torch
logits = torch.tensor([[float('-inf'), float('1'), float('-inf')]])
criterion = torch.nn.CrossEntropyLoss()
print(criterion(logits, torch.tensor([0])).item())
# probs = torch.softmax(logits, dim=0)  # Can produce NaN
# print(probs)

# user_x = torch.tensor([[[[1, -1], [2, -1], [3, -1]], [[4, -1], [5, -1], [6, -1]]]])
# active_idx = torch.tensor([[1, 2]])

# user_x.unsqueeze(2)



# # def var_max(x, idx):
# #     _, x = torch.max(
# #         x.masked_fill((idx == 0).unsqueeze(-1), float("-inf")),
# #         dim=-2,
# #     )

# #     return x.masked_fill(x == float("-inf"), 0)


# # x = torch.tensor([[[1, 2], [2, 1]], [[1, 3], [3, 1]]])
# # idx = torch.tensor([[1,0],[1,0]])

# # print(var_ma(x, idx))


# def var_max(x, idx):
#     x, _ = torch.max(
#         x.masked_fill((idx == 0).unsqueeze(-1), float("-inf")),
#         dim=-2,
#     )

#     return x.masked_fill(x == float("-inf"), 0)

# print(var_max())




# print(criterion(torch.tensor([[0,0,100,0]], dtype=torch.float64), torch.tensor([2])).item())

# print(torch.argmax(torch.tensor([[0,0,1], [1,0,0]]), axis=1))

# v = torch.arange(0, 20).reshape(1, 4, 5)
# print(v.flatten(star))
# v = v.unsqueeze(2).expand(-1, -1, 2, -1)
# w = torch.arange(0,2).reshape(1, 1, 2, 1).expand(1, 4, -1, -1)
# z = torch.arange(50, 55).unsqueeze(0)
# print(z)
# z = z.reshape(1, 1, 1, z.shape[1]).expand(1, 4, 2, z.shape[1])
# print(z)
# print(v)
# a = torch.cat([v, w, z], dim =3)[:,:,:,0:1]
# print(a.squeeze(3))


user_x = torch.tensor([[[[1, 7], [2, 8], [3, 9]], [[4, 10], [5, 11], [6, 12]]]])
active_idx = torch.tensor([[1, 2], [3,4]])
# print(user_x.shape, active_idx.shape)
# print(
#     user_x.gather(
#         2,
#         active_idx.reshape(*active_idx.shape, 1, 1).expand(-1, -1, -1, user_x.shape[3]),
#     ).squeeze(2)
# )

print(active_idx[:, None, None, :].shape)