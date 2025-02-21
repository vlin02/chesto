import torch
from lookup import load_lookup
from sample import vectorize_input, vectorize_target, batch_inputs
from pymongo import MongoClient
import numpy as np
import multiprocessing as mp
from io import BytesIO


def process_chunk(chunk):
    tot, n = chunk

    client = MongoClient("mongodb://172.31.30.235:27017")
    db = client.get_database("chesto")
    device = torch.device("cpu")

    results = db.replays.aggregate(
        [
            {"$match": {"uploadtime": {"$mod": [tot, n]}}},
            {"$project": {"steps": 1}},
            {"$unwind": "$steps"},
            {"$match": {"steps": {"$ne": None}}},
        ],
    )

    lookup = load_lookup(db, device)

    inputs = []
    targets = []
    i = 0
    for x in results:
        step = x["steps"]

        inputs.append(vectorize_input(step["observation"], step["options"], lookup))
        targets.append(vectorize_target(step))
        i += 1

    for x in db.replays.find({"uploadtime": {"$mod": [tot, n]}}, {"$project": {"steps": 1}}):
        steps = x["steps"]

        with BytesIO() as stream:
            np.savez_compressed(stream, **batch_inputs(inputs), target=np.stack(targets))
            binary_data = stream.getvalue()
            db.samples.insert_one({"bin": binary_data, "_id": i})
            print(chunk)


client = MongoClient("mongodb://172.31.30.235:27017")
db = client.get_database("chesto")

batches = [[]]
for x in (
    db.replays.find({}, {"rating": 1, "_id": 1}).sort("rating", 1).batch_size(1000)
):
    batches[-1].append(x)
    if len(batches[-1]) == 100:
        batches.append([])

with mp.Pool(30) as pool:
    results = pool.map(process_chunk, [batch for batch in batches])
