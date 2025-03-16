import torch


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


def get_lookup(db, device):
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


def vectorize_state(lookup, state, device):
    ally = state["ally"]
    foe = state["foe"]
    opt = state["option"]

    user_enc = torch.zeros(2, 6, user_enc_dim)
    active_idx = torch.zeros(2, dtype=torch.long)
    move_choice_idx = torch.zeros(4, dtype=torch.long)
    party_enc = torch.zeros(2, party_enc_dim)

    for i, party in enumerate([ally, foe]):
        team = party["team"]
        active_idx[i] = list(team.keys()).index(party["active"])
        user_enc[i][: len(team)] = torch.tensor([*team.values()], device=device)
        party_enc[i] = torch.tensor(party["x"], device=device)

    move_mask = torch.zeros((4, 2), device=device)
    switch_mask = torch.zeros(6, device=device)

    if opt:
        tera = opt["tera"]
        moves = opt["moves"]
        switches = opt["switches"]

        for i, move in enumerate(moves):
            move_choice_idx[i] = lookup["move_idx"][move]
            for j in range(2):
                if j == 1 and (not tera):
                    continue
                move_mask[i] = 1

        for i, species in enumerate(ally["team"].keys()):
            if species in switches:
                switch_mask[i] = 1

        return dict(
            party_enc=party_enc,
            user_enc=user_enc,
            active_idx=active_idx,
            move_mask=move_mask,
            switch_mask=switch_mask,
            move_choice_idx=move_choice_idx,
        )


def batch_states(inputs):
    x = {k: torch.stack([x[k] for x in inputs]) for k in INPUT_KEYS}
    x["batch_idx"] = torch.arange(len(inputs))
    return x
