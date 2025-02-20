import numpy as np
from pymongo import MongoClient
from io import BytesIO
from lookup import load_lookup
from net import Net
import time
import torch

client = MongoClient("mongodb://172.31.30.235:27017")
db = client.get_database("chesto")
device = torch.device("cuda")

start_time = time.time()
lookup = load_lookup(db, )

net = Net(lookup)

for block in db.samples.find():
    x = np.load(BytesIO(block["bin"]))
    print(x["target"])
    net(x)

print(time.time() - start_time)
