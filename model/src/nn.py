import torch
from torch import nn

user_enc_dim = 28
move_feat_dim = 26
move_enc_dim = move_feat_dim


def get_lookup(db, device):
    lookup = {}
    moves = list(db["moves"].find())
    move_idx = {x["name"]: x["i"] for x in moves}
    move_idx["Recharge"] = 0
    move_enc = torch.zeros(1000, move_enc_dim, device=device)

    for move in moves:
        move_enc[move["i"]] = torch.tensor(move["x"], device=device)

    lookup["move_idx"] = move_idx
    lookup["move_enc"] = move_enc
    return lookup


def process_input(lookup, state, device):
    ally = state["ally"]
    foe = state["foe"]
    opt = state["option"]

    user_enc = torch.zeros(2, 6, user_enc_dim)
    active_idx = torch.zeros(2, dtype=torch.long)
    move_choice_idx = torch.zeros(4, dtype=torch.long)

    for i, party in enumerate([ally, foe]):
        team = party["team"]
        active_idx[i] = list(team.keys()).index(party["active"])
        user_enc[i][: len(team)] = torch.tensor([*team.values()], device=device)

    move_mask = torch.zeros((4, 2), device=device)
    switch_mask = torch.zeros(6, device=device)

    if opt:
        tera = opt["tera"]
        moves = opt["moves"]
        switches = opt["switches"]

        for i, move in enumerate(moves):
            move_choice_idx[i] = lookup["move_idx"][move]
            for j in range(2):
                if j == 1 and (not tera):
                    continue
                move_mask[i] = 1

        for i, species in enumerate(ally["team"].keys()):
            if species in switches:
                switch_mask[i] = 1

        return dict(
            user_enc=user_enc,
            active_idx=active_idx,
            move_mask=move_mask,
            switch_mask=switch_mask,
            move_choice_idx=move_choice_idx,
        )


def batch_inputs(inputs):
    x = {
        k: torch.stack([x[k] for x in inputs])
        for k in [
            "user_enc",
            "active_idx",
            "move_mask",
            "switch_mask",
            "move_choice_idx",
        ]
    }
    x["batch_idx"] = torch.arange(len(inputs))
    return x


class NN(nn.Module):
    def __init__(self, lookup):
        super().__init__()

        self.lookup = lookup
        self.move_embed_block = nn.Sequential(
            nn.Linear(move_enc_dim, 64), nn.Tanh(), nn.Linear(64, 16)
        )
        self.user_block = nn.Sequential(
            nn.Linear(user_enc_dim, 64), nn.Tanh(), nn.Linear(64, 32)
        )
        self.battle_block = nn.Sequential(
            nn.Linear(32 * 2, 64), nn.Tanh(), nn.Linear(64, 32)
        )
        self.move_logit_block = nn.Sequential(
            nn.Linear(32 + 16 + 1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.switch_logit_block = nn.Sequential(
            nn.Linear(32 + 32, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.register_buffer("party_idx", torch.arange(2, dtype=torch.long).view(1, 2))
        self.register_buffer(
            "tera_flag", torch.arange(2).view(1, 1, 2, 1).expand(-1, 4, -1, -1)
        )
        self.critic = nn.Sequential()

    def forward(self, x):
        user_enc = x["user_enc"]
        active_idx = x["active_idx"]
        move_mask = x["move_mask"]
        switch_mask = x["switch_mask"]
        move_choice_idx = x["move_choice_idx"]
        batch_idx = x["batch_idx"]
        batch_dim = batch_idx.size(0)

        move_choice_emb = self.move_embed_block(
            self.lookup["move_enc"][move_choice_idx]
        )

        user_emb = self.user_block(user_enc)

        battle_emb = self.battle_block(
            user_emb[
                batch_idx.view(1, batch_dim).expand(-1, 2),
                self.party_idx.expand(batch_dim, -1),
                active_idx,
            ].reshape(batch_dim, -1)
        )

        move_logit = self.move_logit_block(
            torch.cat(
                [
                    move_choice_emb.view(batch_dim, 4, 1, 16).expand(-1, -1, 2, -1),
                    battle_emb.view(batch_dim, 1, 1, 32).expand(-1, 4, 2, -1),
                    self.tera_flag.expand(batch_dim, -1, -1, -1),
                ],
                dim=-1,
            )
        ).squeeze(-1)

        switch_logit = self.switch_logit_block(
            torch.cat(
                [user_emb[:, 0], battle_emb.view(batch_dim, 1, 32).expand(-1, 6, -1)],
                dim=-1,
            )
        ).squeeze(-1)

        move_logit += (move_mask - 1) * 1e9
        switch_logit += (switch_mask - 1) * 1e9

        return torch.cat([move_logit.flatten(1), switch_logit], dim=-1)
