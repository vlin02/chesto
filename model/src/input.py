import torch
from base64 import b64decode


user_enc_dim = 28
move_feat_dim = 26
move_enc_dim = move_feat_dim
party_enc_dim = 7

INPUT_KEYS = [
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
    move_idx = {x["name"]: x["i"] for x in moves}
    move_idx["Recharge"] = 0
    move_enc = torch.zeros(1000, move_enc_dim, device=device)

    for move in moves:
        move_enc[move["i"]] = torch.tensor(move["x"], device=device)

    lookup["move_idx"] = move_idx
    lookup["move_enc"] = move_enc
    return lookup


def decode_state(state, device):
    x = {}
    for _k, k, dtype, shape in [
        ("userEnc", "user_enc", torch.float32, (2, 6, user_enc_dim)),
        ("partyEnc", "party_enc", torch.float32, (2, party_enc_dim)),
        ("activeIdx", "active_idx", torch.int32, (2,)),
        ("moveMask", "move_mask", torch.int32, (4, 2)),
        ("switchMask", "switch_mask", torch.int32, (6,)),
        ("moveChoiceIdx", "move_choice_idx", torch.int32, (4,)),
    ]:
        x[k] = (
            torch.frombuffer(bytearray(b64decode(state[_k])), dtype=dtype)
            .reshape(*shape)
            .to(device)
        )
    return x


# def batch_states(inputs, device):
#     x = {k: torch.stack([x[k] for x in inputs]) for k in FIELDS.keys()}
#     x["batch_idx"] = torch.arange(len(inputs), device=device)
#     return x
