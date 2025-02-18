from pymongo import MongoClient
import torch
from sample import vectorize_input, vectorize_target, batch_inputs
from lookup import load_lookup
from model import Net
from db import tensor_to_binary, binary_to_tensor
import numpy as np
import time
# @profile
# def main():


#     torch.manual_seed(42)

#     device = torch.device("cuda")
#     lookup = load_lookup(db, device)


#     nn = torch.compile(Net(lookup).to(device), mode="reduce-overhead")


#     batches = list(load())
#     print("here")
#     for x, y in batches:
#         nn(x)
#         print(0)
# main()


# np.save("inputs", np.array(inputs, dtype=object))
# print(np.load("inputs.npy", allow_pickle=True))

# torch.save(inputs, "inputs.pt")
# print("here")
# inputs = torch.load("inputs.pt", weights_only=True)
# print(inputs[0]["ability_idx"].device)
# # print(len(inputs))
# print(nn.forward(batch(inputs)))



# def load_samples(db: MongoClient):
#     return db.replays.aggregate(
#         [
#             {"$limit": 1000},
#             {"$project": {"samples": 1}},
#             {"$unwind": "$samples"},
#             {"$match": {"samples": {"$ne": None}}},
#         ],
#     )


client = MongoClient("mongodb://localhost:27017")
db = client.get_database("chesto")

# device = torch.device("cuda")
# results = load_samples(db)
# lookup = load_lookup(db, device)


# samples = []
# i = 0
# for _ in range(10000):
#     res = results.next()["samples"]
#     obs = res["observation"]
#     options = res["options"]
#     choice = res["choice"]

#     input = vectorize_input(lookup, obs, options)
#     target = vectorize_target(obs, options, choice)
#     if np.sum(target) != 1:
#         print(res)
#         raise ""

#     samples.append(dict(input=tensor_to_binary(input), target=tensor_to_binary(target)))
#     i += 1
#     if i % 100 == 0:
#         print(i)

# import uuid

# i = 0
# samples = []
# for s in db.samples.find().limit(50000):
#     v = binary_to_tensor(s["input"]).item()
#     # v["target"] = binary_to_tensor(s["target"])
#     samples.append(v)
#     i += 1
#     if i % 100 == 0:
#         print(i)



# v = batch_inputs(samples)

# # # np.savez_compressed("all5", **v)


# torch.save(v, "all1.npy")

# samples = np.array(samples)
# print(samples.shape[0], len(samples.shape))
# # print(samples[1]["item_idx"])
# fp = np.memmap('data.mmap', dtype=object, mode='w+', shape=samples.shape)
# fp[:] = samples
# fp.flush()

# fp1 = np.memmap('data.mmap', dtype=object, mode='r', shape=(10,))
# print(fp1[:][9]["move_set_idx"])

# np.savez_compressed("data", **{k: v for k, v in samples})

# # Load with mmap


# print(v1["move_set_x"].shape)


# torch.manual_seed(42)


# # nn = Net(lookup).to(device)

# print("started")
# start_time = time.time()
# v1 = torch.load("all1.npy", map_location="cuda")
# x = np.load("all5.npz")

# smove_set_idx
# smove_set_x
# smove_pool_idx
# smove_pool_x
# smove_lookup_idx
# smove_lookup_x
# sability_idx
# sitem_idx
# sitem_mask
# sitem_lookup_idx
# suser_x
# suser_mask
# sside_x
# sactive_idx
# sbattle_x
# smove_option_idx
# smove_option_x
# smove_option_mask
# sswitch_option_mask
# print(time.time() - start_time)

# for i in range(100):
#   print(nn(v1, 500 * i, 500 * i + 500)[0][0])
# print(time.time() - start_time)