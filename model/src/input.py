from aiohttp import ClientSession
from attr import dataclass
import torch
import numpy as np

user_enc_dim = 52
move_enc_dim = 30
team_enc_dim = 9
type_count = 20


STATE_FIELDS = [
    "user_enc",
    "active_idx",
    "move_mask",
    "switch_mask",
    "move_choice_idx",
    "team_enc",
]

@dataclass
class Lookup:
    move_enc: any

async def load_lookup(session: ClientSession, device):
    res = await session.get("/moves")
    moves = (await res.json())["moves"]
    
    lookup = {}
    move_enc = torch.zeros(1000, move_enc_dim, device=device)

    for move in moves:
        move_enc[move["num"]] = torch.tensor(move["x"], device=device)

    lookup["move_enc"] = move_enc
    return Lookup(move_enc=move_enc)

def decode_states(states, device):
    x = {}
    N = len(states)
    for _k, k, dtype, shape in [
        ("userEnc", "user_enc", np.float32, (N, 7, user_enc_dim)),
        ("teamEnc", "team_enc", np.float32, (N, 2, team_enc_dim)),
        ("activeIdx", "active_idx", np.int32, (N, 2,)),
        ("moveMask", "move_mask", np.int32, (N, 4, 2)),
        ("switchMask", "switch_mask", np.int32, (N, 6,)),
        ("moveChoiceIdx", "move_choice_idx", np.int32, (N, 4,)),
    ]:
        a = b''.join([x[_k] for x in states])
        a = bytearray(a)
        a = np.frombuffer(a, dtype=dtype)
        a = a.reshape(shape)
        x[k] = torch.from_numpy(a).to(device)
    return x
