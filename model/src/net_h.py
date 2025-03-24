import torch
from attr import dataclass
from torch import nn

USER_ENC_DIM = 52
MOVE_ENC_DIM = 30
TEAM_ENC_DIM = 9
TYPE_COUNT = 20
HIDDEN_DIM = 64


@dataclass
class Config:
    user_enc_dim: int = USER_ENC_DIM
    move_enc_dim: int = MOVE_ENC_DIM
    team_enc_dim: int = TEAM_ENC_DIM
    n_types: int = TYPE_COUNT
    hidden_dim: int = HIDDEN_DIM


class Net(nn.Module):
    def __init__(self, lookup, c: Config):
        super().__init__()
        self.lookup = lookup
        self.move_embed_block = nn.Sequential(
            nn.Linear(c.move_enc_dim, c.hidden_dim),
            nn.Tanh(),
        )
        self.user_embed_block = nn.Sequential(
            nn.Linear(c.user_enc_dim, c.hidden_dim),
            nn.Tanh(),
        )
        self.user_matchup_block = nn.Sequential(
            nn.Linear(2 * c.hidden_dim, c.hidden_dim), nn.Tanh()
        )
        self.team_embed_block = nn.Sequential(nn.Linear(c.team_enc_dim, c.hidden_dim), nn.Tanh())
        self.move_logits_block = nn.Sequential(
            nn.Linear(4 * c.hidden_dim + 1, c.hidden_dim),
            nn.Tanh(),
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.Tanh(),
            nn.Linear(c.hidden_dim, 1),
        )
        self.register_buffer(
            "tera_flag", torch.arange(2).view(1, 1, 2, 1).expand(-1, 4, -1, -1)
        )
        self.switch_logits_block = nn.Sequential(
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.Tanh(),
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.Tanh(),
            nn.Linear(c.hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(c.hidden_dim * 3, c.hidden_dim),
            nn.Tanh(),
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.Tanh(),
            nn.Linear(c.hidden_dim, 1),
        )
        self.c = c

    def forward(self, x):
        user_enc = x["user_enc"]
        move_mask = x["move_mask"]
        switch_mask = x["switch_mask"]
        move_choice_idx = x["move_choice_idx"]
        team_enc = x["team_enc"]
        batch_size = user_enc.shape[0]

        move_choice_emb = self.move_embed_block(self.lookup.move_enc[move_choice_idx])
        team_emb = self.team_embed_block(team_enc)

        user_emb = self.user_embed_block(user_enc)
        foe_active = user_emb[:, 6]

        match_up_emb = self.user_matchup_block(
            torch.cat(
                [
                    user_emb[:, :6],
                    foe_active.view(batch_size, 1, self.c.hidden_dim).expand(-1, 6, -1),
                ],
                dim=-1,
            )
        )

        move_logits = self.move_logits_block(
            torch.cat(
                [
                    match_up_emb[:, 0].view(batch_size, 1, 1, -1).expand(-1, 4, 2, -1),
                    team_emb.view(batch_size, 1, 1, -1).expand(-1, 4, 2, -1),
                    move_choice_emb.view(batch_size, 4, 1, self.c.hidden_dim).expand(
                        -1, -1, 2, -1
                    ),
                    self.tera_flag.expand(batch_size, -1, -1, -1),
                ],
                dim=-1,
            )
        ).squeeze(-1)
        switch_logits = self.switch_logits_block(match_up_emb).squeeze(-1)

        base_logits = move_logits, switch_logits

        move_logits = move_logits + (move_mask - 1) * 1e9
        switch_logits = switch_logits + (switch_mask - 1) * 1e9
        logits = torch.cat([move_logits.flatten(1), switch_logits], dim=-1)

        value = self.critic(
            torch.cat([match_up_emb[:, 0], team_emb.view(batch_size, -1)], dim=-1)
        ).squeeze(-1)

        return logits, value, base_logits
