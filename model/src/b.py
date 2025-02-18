from pymongo import MongoClient
import torch
from vec import vectorize_input, vectorize_target, batch_inputs
from lookup import load_lookup
from model import Net
from db import tensor_to_binary, binary_to_tensor
import numpy as np

def load_samples(db: MongoClient):
    return db.replays.aggregate(
        [
            {"$limit": 1000},
            {"$project": {"samples": 1}},
            {"$unwind": "$samples"},
            {"$match": {"samples": {"$ne": None}}},
        ],
    )


client = MongoClient("mongodb://localhost:27017")
db = client.get_database("chesto")

torch.manual_seed(42)

device = torch.device("cuda")
lookup = load_lookup(db, device)

results = load_samples(db)
nn = Net(lookup).to(device)

def load():
    inputs = []
    targets = []
    for s in db.samples.find().limit(40000):
        inputs.append(binary_to_tensor(s["input"]).item())
        targets.append(binary_to_tensor(s["target"]))

        if len(inputs) == 5000:
            yield (
                batch_inputs(inputs, device),
                torch.from_numpy(np.stack(targets)).to(device),
            )
            inputs = []

for x, y in load():
    nn.forward(x)
    print(0)
    
# np.save("inputs", np.array(inputs, dtype=object))
# print(np.load("inputs.npy", allow_pickle=True))

# torch.save(inputs, "inputs.pt")
# print("here")
# inputs = torch.load("inputs.pt", weights_only=True)
# print(inputs[0]["ability_idx"].device)
# # print(len(inputs))
# # print(nn.forward(batch(inputs)))
