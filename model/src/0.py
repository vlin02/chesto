import torch
import h5py
from lookup import load_lookup
from net import Net
from sample import SCHEMA
from pymongo import MongoClient
from pprint import pprint
import numpy as np

KEYS = list(SCHEMA.keys())


def train():
    client = MongoClient("mongodb://172.31.30.235:27017")
    db = client.get_database("chesto")

    device = torch.device("cuda:0")
    lookup = load_lookup(db, device)
    net = Net(lookup).to(device)
    net.load_state_dict(torch.load("__tmp/net-1900-drop"))
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    lo, hi = 0, 2359987

    with h5py.File("__tmp/sample10.hdf5", "r") as f:
        if False:
            batch_size = 500
            tot_batches = (hi - lo) // batch_size
            loaded = {}
            for k in KEYS:
                print(k)
                loaded[k] = torch.from_numpy(f[k][:]).to(device)

            batches = []
            for j in range(tot_batches):
                batches.append(
                    {
                        k: loaded[k][lo + j * batch_size : lo + (j + 1) * batch_size]
                        for k in KEYS
                    }
                )

            for batch_i in range(10000):
                print("BATCH", batch_i)
                for j in range(tot_batches):
                    x = batches[j]

                    optimizer.zero_grad()
                    logits = net(x)
                    loss = criterion(logits, torch.argmax(x["target"], axis=1))
                    loss.backward()
                    optimizer.step()
                    if j % 100 == 0:
                        print(j, loss.item())

                torch.save(net.state_dict(), "__tmp/net-1900-drop")

        else:
            batch_size = 5000
            tot_batches = (hi - lo) // batch_size
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
