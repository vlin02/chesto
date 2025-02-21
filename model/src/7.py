import torch
import h5py
from lookup import load_lookup
from net import Net
from sample import SCHEMA
from pymongo import MongoClient
from pprint import pprint

KEYS = list(SCHEMA.keys())

def train():
    client = MongoClient("mongodb://172.31.30.235:27017")
    db = client.get_database("chesto")

    device = torch.device("cuda:0")
    lookup = load_lookup(db, device)
    net = Net(lookup).to(device)
    net.load_state_dict(torch.load("net-7M-v2"))
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.005)
    criterion = torch.nn.CrossEntropyLoss()

    lo, hi = 0, 786632
    batch_size = 2000
    tot_batches = (hi - lo) // batch_size

    with h5py.File("sample6.hdf5", "r") as f:
        pprint(f["move_option_mask"][:100].tolist())
        return

        for batch_i in range(100):
            print("BATCH", batch_i)
            for j in range(tot_batches):
                x = {
                    k: torch.from_numpy(
                        f[k][lo + j * batch_size : lo + (j + 1) * batch_size]
                    ).to(device)
                    for k in KEYS
                }

                optimizer.zero_grad()
                logits = net(x)
                loss = criterion(logits, torch.argmax(x["target"], axis=1))
                loss.backward()
                optimizer.step()
                print(j, loss.item())

            torch.save(net.state_dict(), "net-7M-v2")

        net.eval()
        with torch.no_grad():
            for i in range(tot_batches):
                x = {
                    k: torch.from_numpy(
                        f[k][lo + i * batch_size : lo + (i + 1) * batch_size]
                    ).to(device)
                    for k in KEYS
                }

                logits = net(x)
                _, pred = torch.max(logits, 1)

                correct = (pred == torch.argmax(x["target"], axis=1)).sum().item()
                acc = correct / batch_size
                print(acc)


train()
