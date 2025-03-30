import torch
from torch import nn

from config import Config
from state import Lookup


class Net(nn.Module):
    def __init__(self, lookup: Lookup, c: Config):
        super().__init__()
        self.lookup = lookup

        self.move_embed_block = nn.Sequential(
            nn.Linear(c.move_feat_dim, c.hidden_dim),
            nn.Tanh(),
        )
        self.user_embed_block = nn.Sequential(
            nn.Linear(c.user_feat_dim, c.hidden_dim),
            nn.Tanh(),
        )
        self.party_embed_block = nn.Sequential(nn.Linear(c.party_feat_dim, c.hidden_dim), nn.Tanh())

        self.user_matchup_block = nn.Sequential(nn.Linear(2 * c.hidden_dim, c.hidden_dim), nn.Tanh())
        
        self.register_buffer("tera_flag", torch.arange(2).view(1, 1, 2, 1).expand(-1, 4, -1, -1))
        self.move_logits_block = nn.Sequential(
            nn.Linear(4 * c.hidden_dim + 1, c.hidden_dim),
            nn.Tanh(),
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.Tanh(),
            nn.Linear(c.hidden_dim, 1),
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
        user_feat = x["user_feat"]
        move_mask = x["move_mask"]
        switch_mask = x["switch_mask"]
        move_choice_idx = x["move_choice_idx"]
        party_feat = x["party_feat"]
        batch_size = user_feat.shape[0]

        move_choice_emb = self.move_embed_block(self.lookup.move_feat[move_choice_idx])
        party_emb = self.party_embed_block(party_feat)

        user_emb = self.user_embed_block(user_feat)
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
                    party_emb.view(batch_size, 1, 1, -1).expand(-1, 4, 2, -1),
                    move_choice_emb.view(batch_size, 4, 1, self.c.hidden_dim).expand(-1, -1, 2, -1),
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

        value = self.critic(torch.cat([match_up_emb[:, 0], party_emb.view(batch_size, -1)], dim=-1)).squeeze(-1)

        return logits, value, base_logits
