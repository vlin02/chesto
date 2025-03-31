import torch
from torch import nn

from lookup import Lookup
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, lookup: Lookup):
        super().__init__()
        c = lookup.c
        self.move_feat = lookup.move_feat
        self.move_type = lookup.move_type

        self.type_block = nn.Sequential(
            nn.Linear(c.n_types, c.hidden_dim),
            nn.Tanh(),
        )

        self.move_block = nn.Sequential(
            nn.Linear(c.hidden_dim + c.move_feat_dim, c.hidden_dim),
            nn.Tanh(),
        )

        self.party_block = nn.Sequential(
            nn.Linear(c.party_feat_dim, c.hidden_dim),
            nn.Tanh(),
        )

        self.user_block = nn.Sequential(
            nn.Linear(2 * c.hidden_dim + c.user_feat_dim, c.hidden_dim),
            nn.Tanh(),
        )

        self.matchup_block = nn.Sequential(nn.Linear(2 * c.hidden_dim, c.hidden_dim), nn.Tanh())

    def type(self, x):
        return self.type_block(x)

    def move(self, idx):
        feat = self.move_feat[idx]
        type = self.move_type[idx]
        type_emb = self.type_block(type)

        return self.move_block(torch.cat([feat, type_emb], dim=-1))

    def party(self, x):
        return self.party_block(x)

    def user(self, feat, type):
        type_emb = self.type_block(type)
        return self.user_block(torch.cat([feat, type_emb.flatten(-2)], dim=-1))

    def matchup(self, ally, foe):
        return self.matchup_block(torch.cat([ally, foe], dim=-1))


class Actor(nn.Module):
    move_feat: torch.Tensor

    def __init__(self, lookup: Lookup):
        super().__init__()

        c = lookup.c
        self.move_feat = lookup.move_feat
        self.encoder = Encoder(lookup)

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

        self.register_buffer("tera_flag", torch.arange(2).view(1, 1, 2, 1).expand(-1, 4, -1, -1))

    def forward(self, x, n):
        user_feat = x["user_feat"]
        user_type = x["user_type"]
        move_choice_idx = x["move_choice_idx"]
        party_feat = x["party_feat"]

        user_emb = self.encoder.user(user_feat, user_type)
        party_emb = self.encoder.party(party_feat)
        move_choice_emb = self.encoder.move(move_choice_idx)
        match_up_emb = self.encoder.matchup(user_emb[:, :6], user_emb[:, 6].view(n, 1, -1).expand(-1, 6, -1))

        move_logits = self.move_logits_block(
            torch.cat(
                [
                    match_up_emb[:, 0].view(n, 1, 1, -1).expand(-1, 4, 2, -1),
                    party_emb.view(n, 1, 1, -1).expand(-1, 4, 2, -1),
                    move_choice_emb.view(n, 4, 1, -1).expand(-1, -1, 2, -1),
                    self.tera_flag.expand(n, -1, -1, -1),
                ],
                dim=-1,
            )
        ).squeeze(-1)
        switch_logits = self.switch_logits_block(match_up_emb).squeeze(-1)

        return move_logits, switch_logits


class Critic(nn.Module):
    def __init__(self, lookup: Lookup):
        super().__init__()
        c = lookup.c

        self.encoder = Encoder(lookup)
        
        self.critic = nn.Sequential(
            nn.Linear(c.hidden_dim * 3, c.hidden_dim),
            nn.Tanh(),
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.Tanh(),
            nn.Linear(c.hidden_dim, 1),
        )

    def forward(self, x, n):
        user_feat = x["user_feat"]
        user_type = x["user_type"]

        party_emb = self.encoder.party(x["party_feat"])

        user_emb = self.encoder.user(user_feat, user_type)
        match_up_emb = self.encoder.matchup(user_emb[:, 0], user_emb[:, 6])

        return self.critic(torch.cat([match_up_emb, party_emb.flatten(1)], dim=-1)).squeeze(-1)


class Agent(nn.Module):
    actor: Actor
    critic: Critic

    def __init__(self, lookup: Lookup):
        super().__init__()

        self.actor = Actor(lookup)
        self.critic = Critic(lookup)

    def forward(self, x, n):
        move_mask = x["move_mask"]
        switch_mask = x["switch_mask"]

        move_logits, switch_logits = self.actor(x, n)
        raw_logits = (move_logits, switch_logits)

        move_logits = move_logits + (move_mask - 1) * 1e9
        switch_logits = switch_logits + (switch_mask - 1) * 1e9
        logits = torch.cat([move_logits.flatten(1), switch_logits], dim=-1)
        dist = torch.distributions.Categorical(F.softmax(logits, dim=1))

        value = self.critic(x, n)

        return dist, value, raw_logits
