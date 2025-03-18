import torch
from pybase64 import b64decode
import numpy as np

user_enc_dim = 28
move_feat_dim = 246
move_embed_dim = 128
move_enc_dim = move_feat_dim + move_embed_dim
party_enc_dim = 7


STATE_FIELDS = [
    "user_enc",
    "active_idx",
    "move_mask",
    "switch_mask",
    "move_choice_idx",
    "party_enc",
]


def load_lookup(db, device):
    lookup = {}
    moves = list(db["moves"].find())
    move_idx = {x["name"]: x["num"] for x in moves}
    move_idx["Recharge"] = 0
    move_enc = torch.zeros(1000, move_enc_dim, device=device)

    for move in moves:
        move_enc[move["num"]] = torch.tensor(move["x"] + move["openai"], device=device)

    lookup["move_idx"] = move_idx
    lookup["move_enc"] = move_enc
    return lookup


def decode_state(states, device):
    x = {}
    N = len(states)
    for _k, k, dtype, shape in [
        ("userEnc", "user_enc", np.float32, (N, 2, 6, user_enc_dim)),
        ("partyEnc", "party_enc", np.float32, (N, 2, party_enc_dim)),
        ("activeIdx", "active_idx", np.int32, (N, 2,)),
        ("moveMask", "move_mask", np.int32, (N, 4, 2)),
        ("switchMask", "switch_mask", np.int32, (N, 6,)),
        ("moveChoiceIdx", "move_choice_idx", np.int32, (N, 4,)),
    ]:
        a = b''.join([b64decode(x[_k]) for x in states])
        a = bytearray(a)
        a = np.frombuffer(a, dtype=dtype)
        a = a.reshape(shape)
        x[k] = torch.from_numpy(a).to(device)
    return x
