import numpy as np
from pymongo import MongoClient
from io import BytesIO
import time
client = MongoClient("mongodb://172.31.30.235:27017")
db = client.get_database("chesto")

start_time = time.time()

for sample in db.samples.find():
  x = np.load(BytesIO(sample["bin"]))
  print(x["move_set_idx"].shape)
  print(x["move_set_x"].shape)
  print(x["move_pool_idx"].shape)
  print(x["move_pool_x"].shape)
  print(x["move_lookup_idx"].shape)
  print(x["move_lookup_x"].shape)
  print(x["ability_idx"].shape)
  print(x["item_idx"].shape)
  print(x["item_mask"].shape)
  print(x["item_lookup_idx"].shape)
  print(x["user_x"].shape)
  print(x["user_mask"].shape)
  print(x["side_x"].shape)
  print(x["active_idx"].shape)
  print(x["battle_x"].shape)
  print(x["move_option_idx"].shape)
  print(x["move_option_x"].shape)
  print(x["move_option_mask"].shape)
  print(x["switch_option_mask"].shape)

print(time.time() - start_time)