from bson import BSON
from fastapi import FastAPI, Request
from pymongo import MongoClient
import torch
import torch.nn.functional as F
import uvicorn

from input import decode_states, load_lookup
from net import NN

app = FastAPI()


torch.set_float32_matmul_precision("high")
device = torch.device("cuda")

DB_URL = "mongodb://admin:4wj62MDCv%25X%5ErU3F@172.31.30.235:27017/"
client = MongoClient(DB_URL)
lookup = load_lookup(client["chesto"], device)

cache = {}


def load(path):
    nn = NN(lookup).to(device)
    state_dict = torch.load(f"__tmp/{path}")
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("_orig_mod.", "")
        new_state_dict[name] = v
    nn.load_state_dict(new_state_dict)
    return nn


@app.post("/predict")
async def predict(request: Request):
    with torch.no_grad():
        body = BSON.decode(await request.body())
        path = body["path"]
        states = body["states"]

        if path not in cache:
            cache[path] = load(path)
        nn = cache[path]

        states = decode_states(states, device)
        logits, v, _ = nn(states)
        probs = F.softmax(logits, dim=1)
        # dist = torch.distributions.Categorical(probs)
        action_ids = torch.max(logits, dim=1).indices

        return [
            dict(probs=probs, action_id=action_id, v=v)
            for probs, action_id, v in zip(
                probs.tolist(), action_ids.tolist(), v.tolist()
            )
        ]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
