import torch
from torch import nn

from input import move_enc_dim, user_enc_dim, party_enc_dim

battle_emb_dim = 64
move_emb_dim = 64
user_emb_dim = 32


class NN(nn.Module):
    def __init__(self, lookup):
        super().__init__()

        self.lookup = lookup
        self.move_embed_block = nn.Sequential(
            nn.Linear(move_enc_dim, 128), nn.Tanh(), nn.Linear(128, move_emb_dim)
        )
        self.user_block = nn.Sequential(
            nn.Linear(user_enc_dim, 64), nn.Tanh(), nn.Linear(64, user_emb_dim)
        )
        self.battle_block = nn.Sequential(
            nn.Linear((32 + party_enc_dim) * 2, 64),
            nn.Tanh(),
            nn.Linear(64, battle_emb_dim),
        )
        self.move_logits_block = nn.Sequential(
            nn.Linear(move_emb_dim + battle_emb_dim + 1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.switch_logits_block = nn.Sequential(
            nn.Linear(user_emb_dim + battle_emb_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.register_buffer("party_idx", torch.arange(2, dtype=torch.long).view(1, 2))
        self.register_buffer(
            "tera_flag", torch.arange(2).view(1, 1, 2, 1).expand(-1, 4, -1, -1)
        )
        self.critic = nn.Sequential(
            nn.Linear(battle_emb_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    # @profile
    def forward(self, x):
        user_enc = x["user_enc"]
        active_idx = x["active_idx"]
        move_mask = x["move_mask"]
        switch_mask = x["switch_mask"]
        move_choice_idx = x["move_choice_idx"]
        party_enc = x["party_enc"]
        batch_dim = user_enc.shape[0]
        batch_idx = torch.arange(batch_dim, device=user_enc.device)

        move_choice_emb = self.move_embed_block(
            self.lookup["move_enc"][move_choice_idx]
        )

        user_emb = self.user_block(user_enc)

        battle_emb = self.battle_block(
            torch.cat(
                [
                    user_emb[
                        batch_idx.view(batch_dim, 1).expand(-1, 2),
                        self.party_idx.expand(batch_dim, -1),
                        active_idx,
                    ].reshape(batch_dim, -1),
                    party_enc.reshape(batch_dim, -1),
                ],
                dim=-1,
            )
        )

        move_logits = self.move_logits_block(
            torch.cat(
                [
                    move_choice_emb.view(batch_dim, 4, 1, move_emb_dim).expand(
                        -1, -1, 2, -1
                    ),
                    battle_emb.view(batch_dim, 1, 1, battle_emb_dim).expand(
                        -1, 4, 2, -1
                    ),
                    self.tera_flag.expand(batch_dim, -1, -1, -1),
                ],
                dim=-1,
            )
        ).squeeze(-1)

        switch_logits = self.switch_logits_block(
            torch.cat(
                [
                    user_emb[:, 0],
                    battle_emb.view(batch_dim, 1, battle_emb_dim).expand(-1, 6, -1),
                ],
                dim=-1,
            )
        ).squeeze(-1)

        move_logits += (move_mask - 1) * 1e9
        switch_logits += (switch_mask - 1) * 1e9

        return torch.cat([move_logits.flatten(1), switch_logits], dim=-1), self.critic(
            battle_emb
        ).squeeze(-1)
