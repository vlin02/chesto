import torch
from torch import nn

MAX_ITEM_NUM = 2500


class Net(nn.Module):
    def __init__(self, lookup):
        super().__init__()
        self.lookup = lookup
        self.fc_item = nn.Linear(256, 128)
        self.fc_ability = nn.Linear(256, 128)
        self.fc_move_slot = nn.Linear(1000, 256)
        self.fc_user = nn.Linear(100, 512)
        self.avg_pool_ability = nn.AvgPool1d()
        self.avg_pool_item = nn.AvgPool1d()
        self.avg_pool_move_slot = nn.AvgPool1d()
        self.max_pool_move = nn.MaxPool1d()
        self.max_pool_user = nn.MaxPool1d()

    def move_slot(self, slot):
        if not slot:
            return torch.zeros(300)

        x = torch.concat(
            slot["f"], self.lookup.moves[slot["move"]] if slot["move"] else 280
        )

        return self.fc_move_slot(x)

    def item(self, name):
        if not name:
            return torch.zeros(300)

        return self.fc_item(self.lookup.items[name])

    def ability(self, name):
        if not name:
            return torch.zeros(300)

        return self.fc_ability(self.lookup.abilities[name])

    def types(self, names):
        x = torch.zeros(20)
        for name in names:
            x[self.lookup.types[name].i] = 1

        return x

    def user(self, user):
        if not user:
            torch.zeros(512)

        lookup = user["lookup"]
        items = user["items"]

        x_move_slots = [self.move_slot(slot) for slot in user["moveSet"]]
        if len(user["movePool"]):
            x_move_slots.append(
                self.avg_pool_move_slot(
                    [self.move_slot(slot) for slot in user["movePool"]]
                )
            )
        x_move_set = self.max_pool_move(x_move_slots)

        x_item = (
            self.avg_pool_item([self.item(name) for name in items])
            if items
            else self.item(None)
        )
        x_ability = self.avg_pool_ability(
            [self.ability(name) for name in user["abilities"]]
        )
        x_types = self.types(user["types"])
        x_tera_type = self.types(user["teraTypes"])

        x = torch.concat(
            user["x"],
            self.move_slot(lookup["disabled"]),
            self.move_slot(lookup["choice"]),
            self.move_slot(lookup["encore"]),
            self.move_slot(lookup["locked"]),
            self.move_slot(lookup["lastMove"]),
            self.item(lookup["lastBerry"]),
            x_move_set,
            x_item,
            x_ability,
            x_types,
            x_tera_type,
        )

        return self.fc_user(x)

    def side(self, side):
        lookup = side["lookup"]
        team = side["team"]

        x_team = {}
        for k in team.keys():
            x_team[k] = self.uerteam[k]

        x = torch.concat(
            side["x"], x_team[lookup["active"]], self.max_pool_user(x_team.values())
        )

        return x

    def obs(self, obs):
        ally = obs["ally"]
        foe = obs["foe"]

        return torch.concat(
            obs["x"],
            self.side(ally),
            self.side(foe),
        )

    def forward(self, obs, opts):
        pass

        # xTeam = {}
        # for species in ally["team"].keys():
        #   xTeam[species] = self.user(ally[])
