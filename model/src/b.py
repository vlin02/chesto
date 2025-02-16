from pymongo import MongoClient
from sample import vectorize_input, vectorize_target


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
sample = samples.next()

obs = sample["observation"]
options = sample["options"]
choice = sample["choice"]

input = vectorize_input(obs, options)
target = vectorize_target(obs, options, choice)

print(input, target)