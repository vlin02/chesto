from dataclasses import dataclass

from aiohttp import ClientSession
import torch


USER_FEAT_DIM = 12
MOVE_FEAT_DIM = 10
PARTY_FEAT_DIM = 10
TYPE_COUNT = 20
HIDDEN_DIM = 256


@dataclass
class Config:
    n_types: int = TYPE_COUNT

    user_feat_dim: int = USER_FEAT_DIM
    move_feat_dim: int = MOVE_FEAT_DIM
    party_feat_dim: int = PARTY_FEAT_DIM

    hidden_dim: int = HIDDEN_DIM


STATE_FIELDS = [
    "user_feat",
    "active_idx",
    "move_mask",
    "switch_mask",
    "move_choice_idx",
    "user_type",
    "party_feat",
]


@dataclass
class Lookup:
    move_feat: torch.Tensor
    move_type: torch.Tensor
    c: Config


async def load_lookup(c: Config, api: ClientSession, device: torch.device):
    res = await api.get("/moves")
    moves = (await res.json())["moves"]

    move_feat = torch.zeros(1000, c.move_feat_dim, device=device)
    move_type = torch.zeros(1000, c.n_types, device=device)

    for move in moves:
        move_feat[move["num"]] = torch.tensor(move["x"], device=device)
        move_type[move["num"]] = torch.tensor(move["type"], device=device)

    return Lookup(move_feat=move_feat, c=c, move_type=move_type)
