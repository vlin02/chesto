import numpy as np
import torch

INPUT_KEYS = [
    "ability_idx",
    "active_idx",
    "battle_x",
    "item_idx",
    "item_lookup_idx",
    "item_mask",
    "move_lookup_idx",
    "move_lookup_x",
    "move_option_idx",
    "move_option_mask",
    "move_option_x",
    "move_pool_idx",
    "move_pool_x",
    "move_set_idx",
    "move_set_x",
    "side_x",
    "switch_option_mask",
    "user_mask",
    "user_x",
]

SCHEMA = dict(
    ability_idx=dict(shape=(2, 6, 3), dtype=np.int64),
    active_idx=dict(shape=(2,), dtype=np.int64),
    battle_x=dict(shape=(9,), dtype=np.float32),
    item_idx=dict(shape=(2, 6, 3), dtype=np.int64),
    item_lookup_idx=dict(shape=(2, 6, 1), dtype=np.int64),
    item_mask=dict(shape=(2, 6), dtype=np.int64),
    move_lookup_idx=dict(shape=(2, 6, 5), dtype=np.int64),
    move_lookup_x=dict(shape=(2, 6, 5, 2), dtype=np.float32),
    move_option_idx=dict(shape=(4,), dtype=np.int64),
    move_option_mask=dict(shape=(4, 2), dtype=np.int64),
    move_option_x=dict(shape=(4, 2), dtype=np.float32),
    move_pool_idx=dict(shape=(2, 6, 10), dtype=np.int64),
    move_pool_x=dict(shape=(2, 6, 10, 2), dtype=np.float32),
    move_set_idx=dict(shape=(2, 6, 4), dtype=np.int64),
    move_set_x=dict(shape=(2, 6, 4, 2), dtype=np.float32),
    side_x=dict(shape=(2, 17), dtype=np.float32),
    switch_option_mask=dict(shape=(6,), dtype=np.int64),
    user_mask=dict(shape=(2, 6), dtype=np.int64),
    user_x=dict(shape=(2, 6, 129), dtype=np.float32),
    target=dict(shape=(14,), dtype=np.int64),
)


def one_hot_types(lookup, types):
    x = np.zeros(lookup["dim"]["n_types"])
    x[[lookup["type_idx"][k] for k in types]] = 1
    return x


def vectorize_input(obs, options, lookup):
    dim = lookup["dim"]

    ability_idx = np.zeros((2, 6, 3), dtype=np.int64)
    active_idx = np.zeros(2, dtype=np.int64)
    battle_x = np.zeros(dim["battle_feat"], dtype=np.float32)
    item_idx = np.zeros((2, 6, 3), dtype=np.int64)
    item_lookup_idx = np.zeros((2, 6, 1), dtype=np.int64)
    item_mask = np.ones((2, 6), dtype=np.int64)
    move_lookup_idx = np.zeros((2, 6, 5), dtype=np.int64)
    move_lookup_x = np.zeros((2, 6, 5, dim["slot_feat"]), dtype=np.float32)
    move_option_idx = np.zeros((4), dtype=np.int64)
    move_option_mask = np.ones((4, 2), dtype=np.int64)
    move_option_x = np.zeros((4, dim["slot_feat"]), dtype=np.float32)
    move_pool_idx = np.zeros((2, 6, 10), dtype=np.int64)
    move_pool_x = np.zeros((2, 6, 10, dim["slot_feat"]), dtype=np.float32)
    move_set_idx = np.zeros((2, 6, 4), dtype=np.int64)
    move_set_x = np.zeros((2, 6, 4, dim["slot_feat"]), dtype=np.float32)
    side_x = np.zeros((2, dim["side_feat"]), dtype=np.float32)
    switch_option_mask = np.ones(6, dtype=np.int64)
    user_mask = np.ones((2, 6), dtype=np.int64)
    user_x = np.zeros((2, 6, dim["user_feat"] + 2 * dim["n_types"]), dtype=np.float32)

    battle_x = np.asarray(obs["x"])

    sides = [obs["ally"], obs["foe"]]
    for i in range(2):
        side = sides[i]
        team = side["team"]
        species = list(team.keys())

        side_x[i] = np.asarray(side["x"])
        active_idx[i] = species.index(side["active"])

        for j in range(6):
            if j < len(species):
                user = team[species[j]]
                move_set = user["moveSet"]
                move_pool = user["movePool"]
                abilities = user["abilities"]
                items = user["items"]
                types = user["types"]
                tera_types = user["teraTypes"]

                user_x[i][j] = np.concatenate(
                    [
                        np.asarray(user["x"]),
                        one_hot_types(lookup, types),
                        one_hot_types(lookup, tera_types),
                    ]
                )

                for k in range(4):
                    if k < len(move_set):
                        slot = move_set[k]
                        a = ["a"]
                        a[0] = lookup["move_idx"][slot["move"]]
                        move_set_idx[i, j, k] = lookup["move_idx"][slot["move"]]
                        v = np.asarray(slot["x"])

                        move_set_x[i, j, k] = v

                for k in range(6):
                    if k < len(move_pool):
                        slot = move_pool[k]
                        move_pool_idx[i, j, k] = lookup["move_idx"][slot["move"]]
                        move_pool_x[i, j, k] = np.asarray(slot["x"])

                for k, ref in enumerate(
                    ["disabled", "choice", "encore", "locked", "lastMove"]
                ):
                    if user[ref]:
                        slot = user[ref]
                        move_lookup_idx[i, j, k] = lookup["move_idx"][slot["move"]]
                        move_lookup_x[i, j, k] = np.asarray(slot["x"])
                
                for k in range(3):
                    if k < len(abilities):
                        ability_idx[i, j, k] = lookup["ability_idx"][abilities[k]]

                for k in range(3):
                    if items and k < len(items):
                        item_idx[i, j, k] = lookup["item_idx"][items[k]]

                if not items:
                    item_mask[i, j] = 0

                for k, ref in enumerate(["lastBerry"]):
                    if user[ref]:
                        item_lookup_idx[i, j, k] = lookup["item_idx"][user[ref]]
            else:
                user_mask[i, j] = 0

    for i in range(4):
        if i < len(options["moves"]):
            slot = options["moves"][i]
            move_option_idx[i] = (
                dim["n_moves"] - 1
                if slot["move"] == "Recharge"
                else lookup["move_idx"][slot["move"]]
            )
            move_option_x[i] = np.asarray(slot["x"])

    for i in range(4):
        for j in range(2):
            if not ((options["canTera"] or j == 0) and i < len(options["moves"])):
                move_option_mask[i][j] = 0

    species = list(obs["ally"]["team"].keys())
    for i in range(6):
        if species[i] not in options["switches"]:
            switch_option_mask[i] = 0

    return dict(
        move_set_idx=move_set_idx,
        move_set_x=move_set_x,
        move_pool_idx=move_pool_idx,
        move_pool_x=move_pool_x,
        move_lookup_idx=move_lookup_idx,
        move_lookup_x=move_lookup_x,
        ability_idx=ability_idx,
        item_idx=item_idx,
        item_mask=item_mask,
        item_lookup_idx=item_lookup_idx,
        user_x=user_x,
        user_mask=user_mask,
        side_x=side_x,
        active_idx=active_idx,
        battle_x=battle_x,
        move_option_idx=move_option_idx,
        move_option_x=move_option_x,
        move_option_mask=move_option_mask,
        switch_option_mask=switch_option_mask,
    )


def vectorize_target(step):
    obs = step["observation"]
    options = step["options"]
    choice = step["choice"]

    move_choice = np.zeros((4, 2), dtype=np.int64)
    switch_choice = np.zeros((6), dtype=np.int64)

    team = obs["ally"]["team"]
    species = list(team.keys())

    if choice["type"] == "move":
        i = [x["move"] for x in options["moves"]].index(choice["move"])
        j = int(choice["tera"])
        move_choice[i][j] = 1

    elif choice["type"] == "switch":
        i = species.index(choice["species"])
        switch_choice[i] = 1

    return np.concatenate([move_choice.flatten(), switch_choice])


def batch_inputs(inputs, device):
    return {k: torch.from_numpy(np.stack([x[k] for x in inputs])).to(device) for k in INPUT_KEYS}


def slice_batch(batch, i, j):
    return {k: batch[k][i:j] for k in INPUT_KEYS}
