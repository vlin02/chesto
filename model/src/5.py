from pymongo import MongoClient
import torch
from lookup import load_lookup
from sample import vectorize_input, vectorize_target
from pprint import pprint
client = MongoClient("mongodb://172.31.30.235:27017")
db = client.get_database("chesto")
device = torch.device("cpu")
lookup = load_lookup(db, device)

samples = []
for x in db.replays.find(
    # {"uploadtime": {"$mod": [tot, n]}, "rating": {"$gt": 2000}}, {"steps": 1}
):
    steps = x["steps"]
    for i, step in enumerate(steps):
        if not step:
            continue

        sample = {}
        obs = step["observation"]
        options = step["options"]
        for k, v in vectorize_input(obs, options, lookup).items():
            sample[k] = v

        sample["target"] = vectorize_target(step)
        pprint(options)
        pprint(step["choice"])
        pprint(sample["move_option_mask"])
        print(step["observation"]["ally"]["team"].keys())
        pprint(sample["switch_option_mask"])
        pprint(sample["target"])
        print()
        # samples.append(sample)
        break
    