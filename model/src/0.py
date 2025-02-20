import torch
from lookup import load_lookup
from sample import vectorize_input, vectorize_target, batch_inputs
from pymongo import MongoClient
import numpy as np
import multiprocessing as mp
from io import BytesIO



def process_chunk(_):
    tot, n = _
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
        if i % 1000 == 0:
            print(n, i)

    with BytesIO() as stream:
        np.savez_compressed(
            stream, **batch_inputs(inputs), target=torch.from_numpy(np.stack(targets))
        )
        binary_data = stream.getvalue()
        db.samples.insert_one({"bin": binary_data, "i": n})


client = MongoClient("mongodb://172.31.30.235:27017")
db = client.get_database("chesto")

db.samples.delete_many({})

with mp.Pool(60) as pool:
    results = pool.map(process_chunk, [(1000, i) for i in range(500)])
