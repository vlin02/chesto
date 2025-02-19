import torch
from lookup import load_lookup
from sample import vectorize_input, vectorize_target
from pymongo import MongoClient

client = MongoClient("mongodb://172.31.30.235:27017")
db = client.get_database("chesto")
device = torch.device("cuda")

results = db.replays.aggregate(
    [
        {"$limit": 1000},
        {"$project": {"steps": 1}},
        {"$unwind": "$steps"},
        {"$match": {"steps": {"$ne": None}}},
    ],
)

lookup = load_lookup(db, device)

for x in results:
    step = x["steps"]

    input = vectorize_input(step, lookup)
    target = vectorize_target(step)
    print(target)
