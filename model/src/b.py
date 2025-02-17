from pymongo import MongoClient
import torch
from vec import vectorize_input, vectorize_target
from lookup import load_lookup
from model import Net


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

lookup = load_lookup(db, torch.get_default_device())

results = load_samples(db)
res = results.next()
sample = res["samples"]

obs = sample["observation"]
options = sample["options"]
choice = sample["choice"]

input = vectorize_input(lookup, obs, options)
target = vectorize_target(obs, options, choice)

nn = Net(lookup)


def batch(inputs):
    to_batch = lambda k: torch.stack([x[k] for x in inputs])

    return dict(
        move_set_idx=to_batch("move_set_idx"),
        move_set_x=to_batch("move_set_x"),
        move_pool_idx=to_batch("move_pool_idx"),
        move_pool_x=to_batch("move_pool_x"),
        move_lookup_idx=to_batch("move_lookup_idx"),
        move_lookup_x=to_batch("move_lookup_x"),
        ability_idx=to_batch("ability_idx"),
        item_idx=to_batch("item_idx"),
        item_mask=to_batch("item_mask"),
        item_lookup_idx=to_batch("item_lookup_idx"),
        user_x=to_batch("user_x"),
        user_mask=to_batch("user_mask"),
        side_x=to_batch("side_x"),
        active_idx=to_batch("active_idx"),
        battle_x=to_batch("battle_x"),
        move_option_idx=to_batch("move_option_idx"),
        move_option_x=to_batch("move_option_x"),
        move_option_mask=to_batch("move_option_mask"),
        switch_option_mask=to_batch("switch_option_mask"),
    )


print(nn.forward(batch([input])))
