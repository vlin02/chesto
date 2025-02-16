import torch
from pymongo import MongoClient
from sample import to_input
from lookup import load_lookup
from pprint import pprint


def load_samples(db):
    return db.replays.aggregate(
        [
            {"$limit": 1000},
            {"$unwind": "$samples"},
            {"$match": {"samples": {"$ne": None}}},
            {"$project": {"samples": 1}},
        ],
    )


client = MongoClient("mongodb://localhost:27017")
db = client.get_database("chesto")

samples = load_samples(db)

device = torch.device("cuda")
lookup = load_lookup(db, device)
# pprint(sample)
# print(to_input(lookup, sample))
print("start")
i = 0
for r in samples:
    try:
        to_input(lookup, r["samples"])
    except Exception as e:
        pprint(r)
        raise e
    
    print(i)
    i+=1
