from flask import Flask, request, jsonify
from sample import vectorize_input, batch_inputs
from pymongo import MongoClient
from lookup import load_lookup
from net import Net
import torch

app = Flask(__name__)

client = MongoClient("mongodb://172.31.30.235:27017")
db = client.get_database("chesto")
device = torch.device("cuda")
lookup = load_lookup(db, device)
net = Net(lookup).to(device)
net.load_state_dict(torch.load("__tmp/net-7M-v2"))


@app.route("/predict", methods=["POST"])
def predict():
    # Get the JSON data from the request
    data = request.get_json()

    options = data["options"]

    obs = data["observation"]
    input = vectorize_input(obs, options, lookup)

    x = batch_inputs([input], device)
    logits = net(x)

    i = int(torch.argmax(logits, dim=1)[0])

    species = list(obs["ally"]["team"].keys())

    def to_choice(i):
        if i < 8:
            j = i // 4
            k = i % 2
            return {"type": "move", "move": options["moves"][j]["move"], "tera": k == 1}
        else:
            return {"type": "switch", "species": species[i - 8]}

    choice = to_choice(i)
    

    return jsonify(choice)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
