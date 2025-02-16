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
            {"$replaceRoot": {"newRoot": "$samples"}},
        ],
    )

client = MongoClient("mongodb://localhost:27017")
db = client.get_database("chesto")

samples = load_samples(db)
sample = samples.next()

device = torch.device('cuda')
lookup = load_lookup(db, device)
pprint(sample)
print(to_input(lookup, sample))
