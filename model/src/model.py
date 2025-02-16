import torch
from torch import nn
from dataclasses import dataclass
from typing import Dict, List
from pymongo import MongoClient
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")


DIMS = dict(
    move_slot_feat=2,
    item_embed=256,
    ability_embed=256,
    move_embed=258 + 256,
    side_feat=17,
    battle_feat=9,
    user_feat=89,
    types=20,
)


@dataclass
class Dex:
    items: Dict[str, List[float]]
    abilities: Dict[str, List[float]]
    moves: Dict[str, List[float]]
    types: Dict[str, List[float]]


def load_dex(db):
    items = {
        f["name"]: torch.tensor(f["desc"]["openai"], device=device)
        for f in db.items.find()
    }
    abilities = {
        f["name"]: torch.tensor(f["desc"]["openai"], device=device)
        for f in db.abilities.find()
    }
    moves = {
        f["name"]: torch.tensor(f["x"] + f["desc"]["openai"], device=device)
        for f in db.moves.find()
    }
    types = {f["name"]: f for f in db.types.find()}

    return Dex(items=items, abilities=abilities, moves=moves, types=types)



class Net(nn.Module):
    def __init__(self, dex):
        super().__init__()
        self.dex = dex

        item_dim = DIMS["item_embed"]
        ability_dim = DIMS["ability_embed"]
        battle_dim = DIMS["battle_feat"] + 2 * (DIMS["side_feat"] + 2 * 512)

        self.no_item = torch.zeros(128, device=device)
        self.no_ability = torch.zeros(128, device=device)
        self.no_move_slot = torch.zeros(128, device=device)

        self.item_block = nn.Sequential(nn.Linear(item_dim, 128), nn.ReLU())
        self.ability_block = nn.Sequential(nn.Linear(ability_dim, 128), nn.ReLU())
        self.move_block = nn.Sequential(
            nn.Linear(DIMS["move_embed"] + DIMS["move_slot_feat"], 128), nn.ReLU()
        )
        self.user_block = nn.Sequential(
            nn.Linear(DIMS["user_feat"] + 9 * 128 + 2 * DIMS["types"], 512), nn.ReLU()
        )
        self.move_opt_block = nn.Sequential(
            nn.Linear(battle_dim + 128 + 1, 512), nn.ReLU(), nn.Linear(512, 1)
        )
        self.switch_opt_block = nn.Sequential(
            nn.Linear(battle_dim + 512, 512), nn.ReLU(), nn.Linear(512, 1)
        )

    def move_slot(self, slot):
        if not slot:
            return self.no_move_slot

        move = slot["move"]
        x = torch.concat(
            [
                torch.tensor(slot["x"], device=device),
                torch.zeros(DIMS["move_embed"], device=device)
                if move == "Recharge"
                else self.dex.moves[slot["move"]],
            ]
        )

        return self.move_slot_block(x)

    def item(self, name):
        if not name:
            return self.no_item
        return self.item_block(self.dex.items[name])

    def ability(self, name):
        if not name:
            return self.no_ability
        return self.ability_block(self.dex.abilities[name])

    def types(self, names):
        x = torch.zeros(20, device=device)
        for name in names:
            x[self.dex.types[name]["num"]] = 1
        return x

    def user(self, user):
        lookup = user["lookup"]
        items = user["items"]

        move_slot_xs = [self.move_slot(slot) for slot in user["moveSet"]]
        if len(user["movePool"]):
            move_slot_xs.append(
                torch.stack([self.move_slot(slot) for slot in user["movePool"]]).mean(
                    dim=0
                )
            )
        move_set_x, _ = torch.stack(move_slot_xs).max(dim=0)

        item_x = (
            torch.stack([self.item(name) for name in items]).mean(dim=0)
            if items
            else self.item(None)
        )
        ability_x = torch.stack(
            [self.ability(name) for name in user["abilities"]]
        ).mean(dim=0)

        types_x = self.types(user["types"])
        tera_type_x = self.types(user["teraTypes"])

        x = torch.concat(
            [
                torch.tensor(user["x"], device=device),
                self.move_slot(lookup["disabled"]),
                self.move_slot(lookup["choice"]),
                self.move_slot(lookup["encore"]),
                self.move_slot(lookup["locked"]),
                self.move_slot(lookup["lastMove"]),
                self.item(lookup["lastBerry"]),
                move_set_x,
                item_x,
                ability_x,
                types_x,
                tera_type_x,
            ]
        )

        return self.user_block(x)

    def move_opt(self, battle_x, slot_x, tera):
        x = torch.concat([battle_x, slot_x, torch.tensor([tera], device=device)])
        return self.move_opt_block(x)

    def switch_opt(self, battle_x, user_x):
        x = torch.concat([battle_x, user_x])
        return self.switch_opt_block(x)

    def side(self, side):
        lookup = side["lookup"]
        team = side["team"]

        team_x = {}
        for k in team.keys():
            team_x[k] = self.user(team[k])

        x = torch.concat(
            [
                torch.tensor(side["x"], device=device),
                team_x[lookup["active"]],
                torch.stack(list(team_x.values())).max(dim=0)[0],
            ]
        )

        return x, team_x

    def forward(self, sample):
        move_x = torch.zeros(2, 6, 10)
        slot_x = torch.zeros(2, 6, 10, 2)
        ability_x = torch.zeros(2, 6, 3)
        ability_n = torch.zeros(2, 6)
        item_x = torch.zeros(2, 6, 3)
        item_n = torch.zeros(2, 6)

        sides = [sample["ally"], sample["foe"]]
        for i in range(2):
            side = sides[i]
            team = side["team"]
            for j in range(team):
                user = team[j]
                move_set = user["moveSet"]
                move_pool = user["movePool"]
                abilities = user["abilities"]
                items = user["items"]

                for k in range(min(len(move_set), 4)):
                    slot = move_set[k]
                    move_x[i][j][k] = slot["move"]
                    slot_x[i][j][k] = slot["x"]

                for k in range(min(len(move_pool), 6)):
                    slot = move_pool[k]
                    move_x[i][j][k + 6] = slot["move"]
                    slot_x[i][j][k + 6] = slot["x"]

                for k in range(min(len(abilities), 3)):
                    ability_x[i][j][k] = abilities[k]
                ability_n[i][j] = min(len(abilities), 3)

                item_n[i][j] = min(len(items), 3)
                for k in range(item_n[i][j]):
                    item_x[i][j][k] = items[k]


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
