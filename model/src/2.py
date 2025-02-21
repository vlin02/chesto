import h5py
from pymongo import MongoClient
import torch
from lookup import load_lookup
from net import Net
from sample import INPUT_KEYS
import numpy as np
from io import BytesIO
import torch.nn.functional as F
from idx import IDX

client = MongoClient("mongodb://172.31.30.235:27017")
db = client.get_database("chesto")

B = 2000
# r = 1600
# i = 0
# d = {}
# for x in db.samples.find({}, {"n": 1, "avg": 1}).sort("avg"):
#     while r < x["avg"]:
#         d[r] = i
#         r += 10
#     print(x["avg"])
#     i += x["n"]
# d[r] = i
# print(d)
# raise ""


SHAPES = [
    ["ability_idx", (2, 6, 3), np.int64],
    ["active_idx", (2,), np.int64],
    ["battle_x", (9,), np.float32],
    ["item_idx", (2, 6, 3), np.int64],
    ["item_lookup_idx", (2, 6, 1), np.int64],
    ["item_mask", (2, 6), np.int64],
    ["move_lookup_idx", (2, 6, 5), np.int64],
    ["move_lookup_x", (2, 6, 5, 2), np.float32],
    ["move_option_idx", (4,), np.int64],
    ["move_option_mask", (4, 2), np.int64],
    ["move_option_x", (4, 2), np.float32],
    ["move_pool_idx", (2, 6, 10), np.int64],
    ["move_pool_x", (2, 6, 10, 2), np.float32],
    ["move_set_idx", (2, 6, 4), np.int64],
    ["move_set_x", (2, 6, 4, 2), np.float32],
    ["side_x", (2, 17), np.float32],
    ["switch_option_mask", (6,), np.int64],
    ["user_mask", (2, 6), np.int64],
    ["user_x", (2, 6, 129), np.float32],
    ["target", (14,), np.int64],
]

def init():
    with h5py.File("sample2.hdf5", "w") as f:
        N = 1
        for k, shape, dtype in SHAPES:
            f.create_dataset(k, [N, *shape], dtype=dtype)

        i = 0
        for block in db.samples.find().sort({"avg": 1}):
            print(block["avg"])
            x = np.load(BytesIO(block["bin"]))
            n = block["n"]
            for k in [*INPUT_KEYS, "target"]:
                f[k][i : i + n] = x[k]
            i += n
            print(i)
    client.close()

def reorder():
    perm = torch.randperm(7338107)
    with h5py.File("sample2.hdf5", "r") as f0:
        with h5py.File("sample3.hdf5", "w") as f:
            for k, *_ in SHAPES:
                print(k)
                f[k] = f0[k][:][perm]

import matplotlib.pyplot as plt

def train():
    device = torch.device("cuda:0")
    lookup = load_lookup(db, device)
    net = Net(lookup).to(device)
    net.load_state_dict(torch.load("net-7M"))
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.005)
    criterion = torch.nn.CrossEntropyLoss()

    lo, hi = 6500000, 7000000
    tot_batches = (hi - lo) // B

    with h5py.File("sample3.hdf5", "r") as f:

        # for b in range(100):
        #     print("BATCH", b)
        #     for j in range(tot_batches):
        #         x = {
        #             k: torch.from_numpy(f[k][lo + j * B : lo + (j + 1) * B]).to(device)
        #             for k in [*KEYS, "target"]
        #         }

        #         optimizer.zero_grad()
        #         logits = net(x)
        #         loss = criterion(logits, torch.argmax(x["target"], axis=1))
        #         loss.backward()
        #         optimizer.step()
        #         print(j, loss.item())
            
        #     torch.save(net.state_dict(), "net-7M")


        torch.set_printoptions(profile="full")
        x = {
            k: torch.from_numpy(f[k][:100000]).to(device)
            for k in [*INPUT_KEYS, "target"]
        }
        v = torch.cat([x["move_option_mask"].flatten(start_dim=1), x["switch_option_mask"]], dim=1)
        print(v.float().mean(dim=0))
        return

        net.eval()

        with torch.no_grad():
            for j in range(tot_batches): 
                i = j
                x = {
                    k: torch.from_numpy(f[k][lo + i * B : lo + (i + 1) * B]).to(device)
                    for k in [*INPUT_KEYS, "target"]
                }
                logits = net(x)
                _, pred = torch.max(logits, 1)

                


                # print(x["target"][0])
                # print(pred == torch.argmax(x["target"], axis=1))
                
                # plt.hist(torch.argmax(x["target"], axis=1).cpu(), bins=range(15))  # adjust number of bins as needed
                # plt.xlabel('Value')
                # plt.ylabel('Frequency')
                # plt.title('Histogram')
                # plt.savefig('line_graph1.png', dpi=300, bbox_inches='tight')

                correct = (pred == torch.argmax(x["target"], axis=1)).sum().item()
                acc = correct / B
                print(acc)

# init()
train()