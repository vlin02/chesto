import torch
from torch import nn
from dataclasses import dataclass
from typing import Dict, List
from pymongo import MongoClient
from line_profiler import profile

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
    items = {f["name"]: torch.tensor(f["desc"]["openai"]) for f in db.items.find()}
    abilities = {
        f["name"]: torch.tensor(f["desc"]["openai"]) for f in db.abilities.find()
    }
    moves = {
        f["name"]: torch.tensor(f["x"] + f["desc"]["openai"]) for f in db.moves.find()
    }

    types = {f["name"]: f for f in db.types.find()}

    return Dex(items=items, abilities=abilities, moves=moves, types=types)


def load_samples(db):
    return db.replays.aggregate(
        [
            {"$unwind": "$steps"},
            {"$match": {"steps.sample": {"$ne": None}}},
            {"$project": {"sample": "$steps.sample"}},
        ],
        cursor={"batchSize": 10000},
    )


class Net(nn.Module):
    def __init__(self, dex):
        super().__init__()
        self.dex = dex

        item_dim = DIMS["item_embed"]
        ability_dim = DIMS["ability_embed"]
        battle_dim = DIMS["battle_feat"] + 2 * (DIMS["side_feat"] + 2 * 512)

        self.no_item = torch.zeros(128)
        self.no_ability = torch.zeros(128)
        self.no_move_slot = torch.zeros(128)

        self.item_block = nn.Sequential(nn.Linear(item_dim, 128), nn.ReLU())
        self.ability_block = nn.Sequential(nn.Linear(ability_dim, 128), nn.ReLU())
        self.move_slot_block = nn.Sequential(
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
    @profile
    def move_slot(self, slot):
        if not slot:
            return self.no_move_slot

        x = torch.concat([torch.tensor(slot["x"]), self.dex.moves[slot["move"]]])

        return self.move_slot_block(x)
    @profile
    def item(self, name):
        if not name:
            return self.no_item

        return self.item_block(self.dex.items[name])
    @profile
    def ability(self, name):
        if not name:
            return self.no_ability

        return self.ability_block(self.dex.abilities[name])
    @profile
    def types(self, names):
        x = torch.zeros(20)
        for name in names:
            x[self.dex.types[name]["num"]] = 1

        return x

    @profile
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
                torch.tensor(user["x"]),
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

    @profile
    def move_opt(self, battle_x, slot_x, tera):
        x = torch.concat([battle_x, slot_x, torch.tensor([tera])])

        return self.move_opt_block(x)

    @profile
    def switch_opt(self, battle_x, user_x):
        x = torch.concat([battle_x, user_x])

        return self.switch_opt_block(x)

    @profile
    def side(self, side):
        lookup = side["lookup"]
        team = side["team"]

        team_x = {}
        for k in team.keys():
            team_x[k] = self.user(team[k])

        x = torch.concat(
            [
                torch.tensor(side["x"]),
                team_x[lookup["active"]],
                torch.stack(list(team_x.values())).max(dim=0)[0],
            ]
        )

        return x, team_x


    @profile
    def forward(self, obs, opt):
        ally = obs["ally"]
        foe = obs["foe"]

        ally_x, ally_team_x = self.side(ally)
        foe_x, _ = self.side(foe)
        battle_x = torch.concat([torch.tensor(obs["x"]), ally_x, foe_x])

        moves = opt["moves"]
        switches = opt["switches"]
        logits = []

        for i in range(4):
            for tera in [0, 1]:
                if i < len(moves) and (("canTera" in opt) or (not tera)):
                    logits.append(self.move_opt(battle_x, moves[i], tera))
                else:
                    logits.append(float("-inf"))

        for i in range(6):
            if i < len(switches):
                logits.append(self.switch_opt(battle_x, ally_team_x[switches[i]]))
            else:
                logits.append(float("-inf"))

        return torch.tensor(logits).float()


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

    return torch.tensor(y).float()



@profile
def main():
    client = MongoClient("mongodb://localhost:27017")
    db = client.get_database("chesto")
    dex = load_dex(db)

    i = 0
    # for _ in db.replays.find({}, {"step.sample": 1}):
    # print()

    model = Net(dex)

    # # for epoch in range()
    for result in load_samples(db):
        sample = result["sample"]
        obs = sample["observer"]
        opt = sample["option"]
        try: 
            model(obs, opt)
        except:
            print(result["_id"])
        # break
        # print(result["sample"].keys())
        # opt = sample["option"]
        # choice = sample["choice"]
        # if to_label(opt, choice).sum() != 1:
        #     raise result["_id"]

        i += 1
        if i % 1000 == 0:
            print(i)
        # break


main()
