import torch
import matplotlib
from pymongo import MongoClient

device = torch.device("cuda")

DB_URL = "mongodb://admin:4wj62MDCv%25X%5ErU3F@172.31.30.235:27017/"

client = MongoClient(DB_URL)

moves = list(client['chesto']['moves'].find())
move_idx = {x["name"]: x["i"] for x in moves}
move_idx["Recharge"] = 0

print(min(x["i"] for x in moves))

move_feat_dim = 5

move_enc_dim = move_feat_dim

move_enc = torch.zeros(1000, move_enc_dim, device=device)

for move in moves:
  move_enc[move["i"]] = move["x"]
