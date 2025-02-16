import torch
from pymongo import MongoClient
from sample import vectorize_input
from lookup import load_lookup
from pprint import pprint


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

samples = load_samples(db)

# device = torch.device("cuda")
lookup = load_lookup(db, torch.device("cpu"))
# pprint(sample)
# print(to_input(lookup, sample))
print("start")
i = 0
for r in samples:
    print(i)
    # try:
    #     to_input(lookup, r["samples"])
    # except Exception as e:
    #     pprint(r)
    #     raise e

    print(i)
    i += 1
