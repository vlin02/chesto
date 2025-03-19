from fastapi import FastAPI, Request
from pymongo import MongoClient
import torch
import torch.nn.functional as F

from input import decode_states, load_lookup
from net import NN

app = FastAPI()


torch.set_float32_matmul_precision("high")
device = torch.device("cuda")

DB_URL = "mongodb://admin:4wj62MDCv%25X%5ErU3F@172.31.30.235:27017/"
client = MongoClient(DB_URL)
lookup = load_lookup(client["chesto"], device)
nn = NN(lookup).to(device)
# nn: NN = torch.compile(nn, mode="reduce-overhead")

state_dict = torch.load("__tmp/0-1742265283.pt")
new_state_dict = {}
for k, v in state_dict.items():
    name = k.replace("_orig_mod.", "")  # Remove the prefix
    new_state_dict[name] = v
nn.load_state_dict(new_state_dict)


@app.post("/predict")
async def predict(request: Request):
    states = await request.json()
    states = decode_states(states, device)
    logits, v = nn(states)
    probs = F.softmax(logits, dim=1)
    dist = torch.distributions.Categorical(probs)
    action_ids = dist.sample()

    return [
        dict(probs=probs, action_id=action_id, v=v)
        for probs, action_id, v in zip(probs.tolist(), action_ids.tolist(), v.tolist())
    ]
