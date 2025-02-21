import torch
from lookup import load_lookup
from sample import vectorize_input, vectorize_target
from pymongo import MongoClient
import numpy as np
from io import BytesIO
from sample import SCHEMA
import multiprocessing as mp
import h5py


def process_samples(chunk):
    tot, n = chunk
    client = MongoClient("mongodb://172.31.30.235:27017")
    db = client.get_database("chesto")
    device = torch.device("cpu")
    lookup = load_lookup(db, device)

    samples = []
    for x in db.replays.find(
        {"uploadtime": {"$mod": [tot, n]}, "rating": {"$gt": 2000}}, {"steps": 1}
    ):
        steps = x["steps"]
        min_i = .5 * len(steps)

        for i, step in enumerate(steps):
            if i < min_i or not step:
                continue

            sample = {}
            obs = step["observation"]
            options = step["options"]
            for k, v in vectorize_input(obs, options, lookup).items():
                sample[k] = v

            sample["target"] = vectorize_target(step)
            samples.append(sample)
    print(chunk)
    return samples


def run():
    with mp.Pool(40) as pool:
        N = 100
        chunks = pool.map(process_samples, [(N, i) for i in range(N)])
        samples = [sample for chunk in chunks for sample in chunk]

        with h5py.File("__tmp/sample7.hdf5", "w") as f:
            for k in SCHEMA.keys():
                print(k)
                f[k] = np.stack([sample[k] for sample in samples])
            print("finishing")
        print(len(samples))


# @profile
# def to_h5py():
#     client = MongoClient("mongodb://172.31.30.235:27017")
#     db = client.get_database("chesto")

#         print("here")

#         i = 0
#         j = 0
#         for x in db.replays.find({"rating": {"$gt": 2200}}, {"samples"}):
#             x = np.load(BytesIO(x["samples"]))

#             n = x["battle_x"].shape[0]
#             for k in SCHEMA.keys():
#                 x[k]
#             i += n
#             j += 1

#             if j % 1000 == 0:
#                 print(j)


# to_h5py()

run()
