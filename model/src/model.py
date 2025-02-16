import torch
from torch import nn
from pymongo import MongoClient


class Net(nn.Module):
    def __init__(self, lookup, device):
        super().__init__()

        self.lookup = lookup
        dim = lookup["dim"]

        self.item_block = nn.Sequential(nn.Linear(dim["item_embed"], 128), nn.ReLU())
        self.ability_block = nn.Sequential(
            nn.Linear(dim["ability_embed"], 128), nn.ReLU()
        )
        self.slot_block = nn.Sequential(
            nn.Linear(dim["move_embed"] + dim["slot"], 128), nn.ReLU()
        )
        self.user_block = nn.Sequential(
            nn.Linear(dim["user_feat"] + 9 * 128 + 2 * dim["n_types"], 512), nn.ReLU()
        )
        self.battle_block = nn.Sequential(
            nn.Linear(dim["battle_feat"] + 2 * (dim["side_feat"] + 2 * 1024), nn.ReLU())
        )
        self.move_opt_block = nn.Sequential(
            nn.Linear(1024 + 128 + 1, 512), nn.ReLU(), nn.Linear(512, 1)
        )
        self.switch_opt_block = nn.Sequential(
            nn.Linear(1024 + 512, 512), nn.ReLU(), nn.Linear(512, 1)
        )

    def slot(self, idx, x):
        lookup = self.lookup
        
        lookup["move_embed"][idx]

    def forward(self, inputs):
        
        

def to_label(opt, choice):
    moves = opt["moves"]
    switches = opt["switches"]

    y = []
    for i in range(4):
        for tera in [0, 1]:
            y.append(
                choice["type"] == "move"
                and i < len(moves)
                and choice["move"] == moves[i]
                and int(choice["tera"]) == tera
            )
    for i in range(6):
        y.append(
            choice["type"] == "switch"
            and i < len(switches)
            and switches[i] == choice["species"]
        )

    return torch.tensor(y, device=device).float()


def main():
    client = MongoClient("mongodb://localhost:27017")
    db = client.get_database("chesto")
    dex = load_dex(db)

    model = Net(dex).to(device)
    model = torch.compile(model, mode="reduce-overhead")

    i = 0
    print("loading")

    v = []
    for sample in load_samples(db):
        v.append(sample)
        # print(sample)
        # obs = sample["obs"]
        # opt = sample["opt"]
        # model(obs, opt)
        # break

        # print(result["sample"].keys())
        # break
        # opt = sample["option"]
        # choice = sample["choice"]
        # if to_label(opt, choice).sum() != 1:
        #     raise result["_id"]

        i += 1
        if i % 100 == 0:
            break
        # break

    start_time = time.perf_counter()
    for s in v:
        model(s["obs"], s["opt"])

    end_time = time.perf_counter()
    print(end_time - start_time)


if __name__ == "__main__":
    main()
