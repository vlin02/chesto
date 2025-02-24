import torch
from torch import nn

from sample import SCHEMA


def var_max(x, idx):
    x, _ = torch.max(
        x.masked_fill((idx == 0).unsqueeze(-1), float("-inf")),
        dim=-2,
    )

    return x.masked_fill(x == float("-inf"), 0)


def var_avg(x, idx):
    mask = (idx != 0).unsqueeze(-1)
    return torch.sum(x * mask, dim=-2) / torch.sum(mask, dim=-2).clamp(min=1)


def var_mask(x, idx):
    return x * idx.clamp(max=1).unsqueeze(-1)


class Net(nn.Module):
    def __init__(self, lookup):
        super().__init__()

        self.lookup = lookup
        self.item_embed_dim = lookup["item_embed"].shape[1]
        self.ability_embed_dim = lookup["ability_embed"].shape[1]
        self.move_embed_dim = lookup["move_embed"].shape[1]
        self.user_x_dim = SCHEMA["user_x"]["shape"][2]
        self.battle_x_dim = SCHEMA["battle_x"]["shape"][0]
        side_x_dim = 17
        user_encode_dim = 512

        self.item_block = nn.Sequential(
            nn.Linear(self.item_embed_dim, 128),
            nn.ReLU(),
        )

        self.ability_block = nn.Sequential(
            nn.Linear(self.ability_embed_dim, 128),
            nn.ReLU(),
        )

        self.slot_block = nn.Sequential(
            nn.Linear(self.move_embed_dim + 2, 128),
            nn.ReLU(),
        )

        self.user_block = nn.Sequential(
            nn.Linear(self.user_x_dim + 10 * 128, user_encode_dim),
            nn.ReLU(),
        )

        self.battle_block = nn.Sequential(
            nn.Linear(
                self.battle_x_dim + 2 * (self.side_x_dim + 2 * user_encode_dim), 1024
            ),
            nn.ReLU(),
        )

        self.move_option_block = nn.Sequential(
            nn.Linear(1024 + 128 + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.switch_option_block = nn.Sequential(
            nn.Linear(1024 + user_encode_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.register_buffer("move_embed", lookup["move_embed"])
        self.register_buffer("item_embed", lookup["item_embed"])
        self.register_buffer("ability_embed", lookup["ability_embed"])
        self.register_buffer("tera_x", torch.arange(2))

    def item(self, idx):
        return self.item_block(self.item_embed[idx])

    def ability(self, idx):
        return self.ability_block(self.ability_embed[idx])

    def slot(self, x, idx):
        return self.slot_block(torch.cat([x, self.move_embed[idx]], dim=-1))

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
        move_option_mask = inputs["move_option_mask"]
        switch_option_mask = inputs["switch_option_mask"]

        move_set_x = var_max(self.slot(move_set_x, move_set_idx), move_set_idx)

        move_pool_x = var_avg(self.slot(move_pool_x, move_pool_idx), move_pool_idx)
        item_x = var_avg(self.item(item_idx), item_idx)
        ability_x = var_avg(self.ability(ability_idx), ability_idx)
        move_lookup_x = var_mask(
            self.slot(move_lookup_x, move_lookup_idx), move_lookup_idx
        )
        item_lookup_x = self.item(item_lookup_idx)

        user_x = self.user_block(
            torch.cat(
                [
                    user_x,
                    move_set_x,
                    move_pool_x,
                    ability_x,
                    item_x,
                    move_lookup_x.flatten(start_dim=3),
                    item_lookup_x.flatten(start_dim=3),
                ],
                dim=3,
            )
        )

        team_x = var_max(user_x, user_mask)

        side_x = torch.cat(
            [
                side_x,
                user_x.gather(
                    2,
                    active_idx[..., None, None].expand(-1, -1, -1, user_x.shape[3]),
                ).squeeze(2),
                team_x,
            ],
            dim=2,
        )

        battle_x = self.battle_block(
            torch.cat([battle_x, side_x.flatten(start_dim=1)], dim=1)
        )

        move_option_x = (
            self.move_option_block(
                torch.cat(
                    [
                        self.slot(move_option_x, move_option_idx)
                        .unsqueeze(2)
                        .expand(-1, -1, 2, -1),
                        self.tera_x[None, None, :, None].expand(
                            battle_x.shape[0], 4, -1, -1
                        ),
                        battle_x[:, None, None, :].expand(-1, 4, 2, -1),
                    ],
                    dim=3,
                )
            ).squeeze(3)
            # .masked_fill(move_option_mask == 0, float("-inf"))
        )

        ally_user_x = user_x[:, 0]

        switch_option_x = (
            self.switch_option_block(
                torch.cat(
                    [ally_user_x, battle_x.unsqueeze(1).expand(-1, 6, -1)],
                    dim=2,
                )
            ).squeeze(2)
            # .masked_fill(switch_option_mask != -1, float("-inf"))
        )

        logits = torch.cat([move_option_x.flatten(start_dim=1), switch_option_x], dim=1)

        return logits
