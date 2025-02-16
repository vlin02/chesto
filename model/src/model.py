import torch
from torch import nn


def var_max(x, idx):
    return torch.max(
        torch.where(idx.unsqueeze(-1) > 0, x, torch.tensor(float("-inf"))),
        dim=-2,
    )


def var_mask(idx):
    return idx.clamp(max=1).unsqueeze(-1)


def var_avg(x, idx):
    mask = var_mask(idx)
    return torch.sum(x * mask, dim=-2) / torch.sum(mask, dim=-2).clamp(min=1)


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
        self.move_option_block = nn.Sequential(
            nn.Linear(1024 + 128 + 1, 512), nn.ReLU(), nn.Linear(512, 1)
        )
        self.switch_option_block = nn.Sequential(
            nn.Linear(1024 + 512, 512), nn.ReLU(), nn.Linear(512, 1)
        )

    def ability(self, idx):
        return self.ability_block(self.lookup["item_embed"][idx])

    def item(self, idx):
        return self.item_block(self.lookup["item_embed"][idx])

    def slot(self, idx, x):
        return self.move_block(torch.cat(self.lookup["move_embed"][idx], x))

    def forward(self, inputs):
        move_set_idx = inputs["move_set_idx"]
        move_set_x = inputs["move_set_x"]
        move_pool_idx = inputs["move_pool_idx"]
        move_pool_x = inputs["move_pool_x"]
        move_lookup_idx = inputs["move_lookup_idx"]
        move_lookup_x = inputs["move_lookup_x"]
        ability_idx = inputs["ability_idx"]
        item_idx = inputs["item_idx"]
        item_lookup_idx = inputs["item_lookup_idx"]
        user_x = inputs["user_x"]
        user_mask = inputs["user_mask"]
        side_x = inputs["side_x"]
        active_idx = inputs["active_idx"]
        battle_x = inputs["battle_x"]
        move_option_idx = inputs["move_option_idx"]
        move_option_x = inputs["move_option_x"]
        action_mask = inputs["action_mask"]

        move_set_x = var_max(self.slot(move_set_idx, move_set_x), move_set_idx)
        move_pool_x = var_avg(self.slot(move_pool_idx, move_pool_x), move_pool_idx)
        move_lookup_x = self.slot(move_lookup_idx, move_lookup_x) * var_mask(
            move_lookup_idx
        )

        ability_x = var_avg(self.ability(ability_idx), ability_idx)
        item_x = var_avg(self.item(item_idx), item_idx)
        item_lookup_x = self.item(item_lookup_idx)

        user_x = self.user_block(
            torch.concat(
                [
                    user_x,
                    move_set_x,
                    move_pool_x,
                    move_lookup_x,
                    ability_x,
                    item_x,
                    item_lookup_x,
                ]
            )
        )
        team_x = var_max(user_x, user_mask)

        side_x = torch.concat([side_x, user_x[active_idx], team_x])
        battle_x = self.battle_block(torch.cat([battle_x, side_x]))

        move_option_x = (
            self.move_option_block(
                torch.concat(
                    [
                        self.slot(move_option_idx, move_option_x).repeat(),
                        battle_x.unsqueeze(-1),
                    ]
                )
            )
            * move_option_mask
        )

        ally_user_x = user_x[0]

        switch_option_x = (
            self.switch_option_block(torch.cat([ally_user_x, battle_x.unsquee(-1)]))
            * switch_option_mask
        )

        logits = torch.cat([move_option_x.flatten(), switch_option_x])
        return logits
