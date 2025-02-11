import torch
from torch import nn
from dataclasses import dataclass
from typing import Dict, List

DIMS = dict(
    item_embed=256,
    move_slot_feat=2,
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


def load_dex(db):
    items = {f["name"]: f["desc"]["mistral"] for f in db.items.find()}
    abilities = {f["name"]: f["desc"]["mistral"] for f in db.abilities.find()}
    moves = {f["name"]: f["x"] + f["desc"]["mistral"] for f in db.moves.find()}

    return Dex(items=items, abilities=abilities, moves=moves)


class Net(nn.Module):
    def __init__(self, dex):
        super().__init__()
        item_dims = DIMS["item_embed"]
        ability_dims = DIMS["ability_embed"]
        move_slot_dims = 256 + DIMS["move_slot_feat"]
        battle_dims = DIMS["battle_feat"] + 2 * (DIMS["side_feat"] + 2 * 512)

        self.no_item = torch.zeros(item_dims)
        self.no_ability = torch.zeros(ability_dims)
        self.no_move_slot = torch.zeros(move_slot_dims)

        self.dex = dex
        self.item_mlp = nn.Sequential(nn.Linear(item_dims, 128), nn.ReLU())
        self.ability_mlp = nn.Sequential(nn.Linear(ability_dims, 128), nn.ReLU())
        self.move_mlp = nn.Sequential(nn.Linear(DIMS["move_embed"], 256), nn.ReLU())
        self.move_slot_mlp = nn.Sequential(
            nn.Linear(256 + DIMS["move_slot_feat"], 128), nn.ReLU()
        )
        self.user_mlp = nn.Sequential(
            nn.Linear(DIMS["user_feat"] + 9 * 128 + 2 * DIMS["types"], 512), nn.ReLU()
        )

        self.move_opt_mlp = nn.Sequential(
            nn.Linear(battle_dims + 128 + 1, 512), nn.ReLU(), nn.Linear(512, 1)
        )
        self.switch_opt_mlp = nn.Sequential(
            nn.Linear(battle_dims + 512, 512), nn.ReLU(), nn.Linear(512, 1)
        )
        self.ability_avg_pool = nn.AvgPool1d()
        self.item_avg_pool = nn.AvgPool1d()
        self.move_avg_pool = nn.AvgPool1d()
        self.move_max_pool = nn.MaxPool1d()
        self.user_max_pool = nn.MaxPool1d()

    def move_slot(self, slot):
        if not slot:
            return torch.zeros(300)

        x = torch.concat(slot["f"], self.dex.moves[slot["move"]])

        return self.move_slot_mlp(x)

    def item(self, name):
        if not name:
            return self.no_item

        return self.item_mlp(self.dex.items[name])

    def ability(self, name):
        if not name:
            return self.no_ability

        return self.ability_mlp(self.dex.abilities[name])

    def types(self, names):
        x = torch.zeros(20)
        for name in names:
            x[self.dex.types[name].i] = 1

        return x

    def user(self, user):
        if not user:
            torch.zeros(512)

        lookup = user["lookup"]
        items = user["items"]

        move_slot_xs = [self.move_slot(slot) for slot in user["moveSet"]]
        if len(user["movePool"]):
            move_slot_xs.append(
                self.move_avg_pool(self.move_slot(slot) for slot in user["movePool"])
            )
        move_set_x = self.move_max_pool(move_slot_xs)

        item_x = (
            self.item_avg_pool([self.item(name) for name in items])
            if items
            else self.item(None)
        )
        ability_x = self.ability_avg_pool(
            [self.ability(name) for name in user["abilities"]]
        )
        types_x = self.types(user["types"])
        tera_type_x = self.types(user["teraTypes"])

        x = torch.concat(
            user["x"],
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
        )

        return self.user_mlp(x)

    def move_opt(self, battle_x, slot_x, tera):
        x = torch.concat(battle_x, slot_x, tera)

        return self.move_opt_mlp(x)

    def switch_opt(self, battle_x, user_x):
        x = torch.concat(
            battle_x,
            user_x,
        )

        return self.switch_opt_mlp(x)

    def side(self, side):
        lookup = side["lookup"]
        team = side["team"]

        team_x = {}
        for k in team.keys():
            team_x[k] = team[k]

        x = torch.concat(
            side["x"], team_x[lookup["active"]], self.user_max_pool(team_x.values())
        )

        return x, team_x

    def forward(self, obs, opts):
        ally = obs["ally"]
        foe = obs["foe"]

        ally_x, ally_team_x = self.side(ally)
        foe_x, _ = self.side(foe)
        battle_x = torch.concat(obs["x"], ally_x, foe_x)

        move_opts = opts["move"]
        switch_opts = opts["switch"]
        logits = []

        for i in range(4):
            for tera in [0, 1]:
                valid = i < len(move_opts) and (opts["tera"] or (not tera))
                logits.append(
                    self.move_opt(battle_x, move_opts[i], tera)
                    if valid
                    else float("-inf")
                )

        for i in range(6):
            species, valid = switch_opts[i]
            logits.append(
                self.switch_opt(battle_x, ally_team_x[species])
                if valid
                else float("-inf")
            )

        return


# client = MongoClient("mongodb://localhost:27017")
# db = client.get_database("chesto")
# print(load_dex(db))
