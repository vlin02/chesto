import numpy as np
import torch
from base64 import b64decode


user_enc_dim = 28
move_feat_dim = 26
move_enc_dim = move_feat_dim
party_enc_dim = 19

INPUT_KEYS = [
    "user_enc",
    "active_idx",
    "move_mask",
    "switch_mask",
    "move_choice_idx",
    "party_enc",
]

FIELDS = dict(
    user_enc=((2, 6, user_enc_dim), torch.float32, "userEnc"),
    party_enc=((party_enc_dim,), torch.float32, "partyEnc"),
    active_idx=((2,), torch.int32, "activeIdx"),
    move_mask=((4, 2), torch.int32, "moveMask"),
    switch_mask=((6,), torch.int32, "switchMask"),
    move_choice_idx=((4,), torch.int32, "moveChoiceIdx"),
)


def load_lookup(db, device):
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


@profile
def vectorize_state(state, device):
    return {
        k: torch.frombuffer(b64decode(state[k1]), dtype=dtype).reshape(dims).to(device)
        for k, (dims, dtype, k1) in FIELDS.items()
    }


def batch_states(inputs, device):
    x = {k: torch.stack([x[k] for x in inputs]) for k in FIELDS.keys()}
    x["batch_idx"] = torch.arange(len(inputs), device=device)
    return x
