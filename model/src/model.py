import torch
from torch import nn

MAX_ITEM_NUM = 2500


class Net(nn.Module):
    def __init__(self, lookup):
        super().__init__()
        self.lookup = lookup
        self.item_fc = nn.Sequential(nn.Linear(256, 128), nn.ReLU())
        self.ability_fc = nn.Sequential(nn.Linear(256, 128), nn.ReLU())
        self.move_slot_fc = nn.Sequential(nn.Linear(1000, 256), nn.ReLU())
        self.user_fc = nn.Sequential(nn.Linear(100, 512), nn.ReLU())
        self.ability_avg_pool = nn.AvgPool1d()
        self.item_avg_pool = nn.AvgPool1d()
        self.move_avg_pool = nn.AvgPool1d()
        self.move_max_pool = nn.MaxPool1d()
        self.user_max_pool = nn.MaxPool1d()

    def move_slot(self, slot):
        if not slot:
            return torch.zeros(300)

        x_move = torch.zeros(298) if slot["move"] == "Recharge" else self.lookup.moves[slot["move"]]

        x = torch.concat(
            slot["f"], 
            x_move
        )

        return self.move_slot_fc(x)

    def item(self, name):
        if not name:
            return torch.zeros(300)

        return self.item_fc(self.lookup.items[name])

    def ability(self, name):
        if not name:
            return torch.zeros(300)

        return self.ability_fc(self.lookup.abilities[name])

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
                self.move_avg_pool(
                    self.move_slot(slot) for slot in user["movePool"]
                )
            )
        x_move_set = self.move_max_pool(x_move_slots)

        x_item = (
            self.item_avg_pool([self.item(name) for name in items])
            if items
            else self.item(None)
        )
        x_ability = self.ability_avg_pool(
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

        return self.user_fc(x)
    
    def move(self, battle, slot):
        return 
    
    def switch(self, battle, user):
        

    def side(self, side):
        lookup = side["lookup"]
        team = side["team"]

        x_team = {}
        for k in team.keys():
            x_team[k] = team[k]

        x = torch.concat(
            side["x"], x_team[lookup["active"]], self.user_max_pool(x_team.values())
        )

        return x, x_team

    def forward(self, obs, opts):
        ally = obs["ally"]
        foe = obs["foe"]

        x_ally, x_ally_team = self.side(ally)
        x_foe, _ = self.side(foe)

        x_battle = torch.concat(obs["x"], x_ally, x_foe)

        x_opts = []
        for i in range(4):
          x_opts.append()
            
        for i in range(6):
          x_opts

        # xTeam = {}
        # for species in ally["team"].keys():
        #   xTeam[species] = self.user(ally[])
