from pymongo import MongoClient
from io import BytesIO
import numpy as np

mongo = MongoClient("mongodb://172.31.30.235:27017")
db = mongo.get_database("chesto")

batches = [[]]
for x in db.samples.find({}):
    x["_id"]
    x1 = np.load(BytesIO(x["bin"]))
    n = x1["battle_x"].shape[0]
    print(n)
    db.samples.update_one({"_id": x["_id"]}, {"$set": {"n": n}})
