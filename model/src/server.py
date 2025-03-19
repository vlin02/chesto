from fastapi import FastAPI
from pymongo import MongoClient
import uvicorn
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
nn: NN = torch.compile(nn, mode="reduce-overhead")
nn.load_state_dict(torch.load("__tmp/0-1742265283.pt"))


@app.post("/predict")
def predict(state):
    states = decode_states([state], device)
    logits, _ = nn(states)
    dist = torch.distributions.Categorical(F.softmax(logits, dim=1))
    action_ids = dist.sample()

    return action_ids[0].item()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
