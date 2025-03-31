from aiohttp import ClientSession
from attr import dataclass
import torch
import numpy as np

from config import Config

STATE_FIELDS = [
    "user_feat",
    "active_idx",
    "move_mask",
    "switch_mask",
    "move_choice_idx",
    "party_feat",
]

@dataclass
class Lookup:
    move_feat: torch.Tensor
    c: Config

async def load_lookup(c: Config, api: ClientSession, device: torch.device):
    res = await api.get("/moves")
    moves = (await res.json())["moves"]
    
    lookup = {}
    move_feat = torch.zeros(1000, c.move_feat_dim, device=device)

    for move in moves:
        move_feat[move["num"]] = torch.tensor(move["x"], device=device)

    lookup["move_feat"] = move_feat
    return Lookup(move_feat=move_feat, c=c)

def decode_states(c: Config, states, device):
    x = {}
    N = len(states)
    for _k, k, dtype, shape in [
        ("userFeat", "user_feat", np.float32, (N, 7, c.user_feat_dim)),
        ("partyFeat", "party_feat", np.float32, (N, 2, c.party_feat_dim)),
        ("activeIdx", "active_idx", np.int32, (N, 2,)),
        ("moveChoiceIdx", "move_choice_idx", np.int32, (N, 4,)),
        ("moveMask", "move_mask", np.int32, (N, 4, 2)),
        ("switchMask", "switch_mask", np.int32, (N, 6,)),
    ]:
        a = b''.join([x[_k] for x in states])
        a = bytearray(a)
        a = np.frombuffer(a, dtype=dtype)
        a = a.reshape(shape)
        x[k] = torch.from_numpy(a).to(device)
    return x
