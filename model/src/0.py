import torch
from lookup import load_lookup
from sample import vectorize_input, vectorize_target
from pymongo import MongoClient

client = MongoClient("mongodb://172.31.30.235:27017")
db = client.get_database("chesto")
device = torch.device("cpu")

results = db.replays.aggregate(
    [
        {"$match": {"uploadtime": {"$mod": [1000, 0]}}},
        {"$project": {"steps": 1}},
        {"$unwind": "$steps"},
        {"$match": {"steps": {"$ne": None}}},
    ],
)

lookup = load_lookup(db, device)

inputs = []
targets = []
for x in results:
    step = x["steps"]

    inputs.append(vectorize_input(step, lookup))
    targets.append(vectorize_target(step))


print(len(inputs))