import torch
from torch import nn

from config import Config
from state import Lookup
import torch.nn.functional as F


class BattleEncoder(nn.Module):
    def __init__(self, lookup: Lookup, c: Config):
        super().__init__()

        self.c = c
        self.lookup = lookup
        self.user_embed_block = nn.Sequential(
            nn.Linear(c.user_feat_dim, c.hidden_dim),
            nn.Tanh(),
        )
        self.party_embed_block = nn.Sequential(nn.Linear(c.party_feat_dim, c.hidden_dim), nn.Tanh())

        self.user_matchup_block = nn.Sequential(nn.Linear(2 * c.hidden_dim, c.hidden_dim), nn.Tanh())

    def forward(self, x):
        user_feat = x["user_feat"]
        party_feat = x["party_feat"]
        batch_size = user_feat.shape[0]

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

        return match_up_emb, party_emb


class Actor(nn.Module):
    def __init__(self, lookup: Lookup, c: Config):
        super().__init__()
        self.lookup = lookup

        self.battle_block = BattleEncoder(lookup, c)
        self.move_embed_block = nn.Sequential(
            nn.Linear(c.move_feat_dim, c.hidden_dim),
            nn.Tanh(),
        )

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

        self.c = c

    def forward(self, x):
        user_feat = x["user_feat"]
        move_choice_idx = x["move_choice_idx"]
        batch_size = user_feat.shape[0]

        move_choice_emb = self.move_embed_block(self.lookup.move_feat[move_choice_idx])

        match_up_emb, party_emb = self.battle_block(x)

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

        return move_logits, switch_logits


class Critic(nn.Module):
    def __init__(self, lookup: Lookup, c: Config):
        super().__init__()
        self.battle_block = BattleEncoder(lookup, c)
        self.critic = nn.Sequential(
            nn.Linear(c.hidden_dim * 3, c.hidden_dim),
            nn.Tanh(),
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.Tanh(),
            nn.Linear(c.hidden_dim, 1),
        )

    def forward(self, x):
        user_feat = x["user_feat"]
        batch_size = user_feat.shape[0]
        match_up_emb, party_emb = self.battle_block(x)

        return self.critic(torch.cat([match_up_emb[:, 0], party_emb.view(batch_size, -1)], dim=-1)).squeeze(-1)


class Agent(nn.Module):
    actor: Actor
    critic: Critic

    def __init__(self, lookup: Lookup, c: Config):
        super().__init__()
        self.actor = Actor(lookup, c)
        self.critic = Critic(lookup, c)

    def forward(self, x):
        move_mask = x["move_mask"]
        switch_mask = x["switch_mask"]

        move_logits, switch_logits = self.actor(x)
        move_logits = move_logits + (move_mask - 1) * 1e9
        switch_logits = switch_logits + (switch_mask - 1) * 1e9
        logits = torch.cat([move_logits.flatten(1), switch_logits], dim=-1)
        dist = torch.distributions.Categorical(F.softmax(logits, dim=1))

        value = self.critic(x)

        raw_logits = (move_logits, switch_logits)

        return dist, value, raw_logits
