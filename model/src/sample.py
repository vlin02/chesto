import torch


def one_hot_types(lookup, types):
    x = torch.zeros(lookup["n_types"])
    x[[lookup.type_idx[k] for k in types]] = 1


def to_input(lookup, sample):
    dim = lookup["dim"]
    battle = sample["battle"]

    options = sample["options"]
    choice = sample["choice"]

    move_set_idx = torch.zeros(2, 6, 4)
    move_set_x = torch.zeros(2, 6, 4, dim["slot"])

    move_pool_idx = torch.zeros(2, 6, 10)
    move_pool_x = torch.zeros(2, 6, 10, dim["slot"])

    move_lookup_idx = torch.zeros(2, 6, 5)
    move_lookup_x = torch.zeros(2, 6, 5, dim["slot"])
    ability_idx = torch.zeros(2, 6)

    item_idx = torch.zeros(2, 6, 3)
    item_mask = torch.ones(2, 6)

    item_lookup_idx = torch.zeros(2, 6, 1)

    user_x = torch.zeros(2, 6, dim["user"] + 2 * dim["types"])
    user_mask = torch.ones(2, 6)

    side_x = torch.zeros(2, dim["side"])
    active_idx = torch.zeros(2)

    battle_x = torch.zeros(dim["battle"])

    move_option_idx = torch.zeros(4)
    move_option_x = torch.zeros(4, dim["slot"])
    action_mask = torch.ones(14)

    target = torch.zeros(14)

    sides = [sample["ally"], sample["foe"]]

    battle_x = torch.tensor(battle["x"])

    for i in range(2):
        side = sides[i]
        team = side["team"]
        species = list(team.keys())

        side_x[i] = torch.tensor(side["x"])
        active_idx[i] = species.index(side["active"])

        for j in range(6):
            if j < len(species):
                user = team[species[j]]
                move_set = user["moveSet"]
                move_pool = user["movePool"]
                abilities = user["abilities"]
                items = user["items"]
                types = user["types"]
                tera_types = user["tera_types"]

                user_x[i][j] = torch.concat(
                    [
                        torch.tensor(user["x"]),
                        one_hot_types(lookup, types),
                        one_hot_types(lookup, tera_types),
                    ]
                )

                for k in range(4):
                    if k < len(move_set):
                        slot = move_set[k]
                        move_set_idx[i][j][k] = lookup.move_idx[slot["move"]]
                        move_set_x[i][j][k] = torch.tensor(slot["x"])

                for k in range(6):
                    if k < len(move_pool):
                        slot = move_pool[k]
                        move_pool_idx[i][j][k] = lookup.move_idx[slot["move"]]
                        move_pool_x[i][j][k] = torch.tensor(slot["x"])

                for k, ref in enumerate(
                    ["disabled", "choice", "encore", "locked", "lastMove"]
                ):
                    if ref in user:
                        slot = user[ref]
                        move_lookup_idx[i][j][k] = lookup.move_idx[slot["move"]]
                        move_lookup_x[i][j][k] = torch.tensor(slot["x"])

                for k in range(3):
                    if k < len(abilities):
                        ability_idx[i][j][k] = lookup.ability_idx[abilities[k]]

                for k in range(3):
                    if items and k < len(items):
                        item_idx[i][j][k] = lookup.item_idx[items[k]]

                if not items:
                    item_mask[i][j] = 0

                for k, ref in enumerate(["lastBerry"]):
                    if ref in user:
                        item_lookup_idx[i][j][k] = lookup.item_idx[user[ref]]
            else:
                user_mask[i][j] = 0

    for i in range(2):
        for j in range(4):
            if options["canTera"] or i == 0 and j < len(options["moves"]):
                slot = options["moves"][j]
                move_option_idx[i * 4 + j] = lookup.move_idx[slot["move"]]
                move_option_x[i * 4 + j] = torch.tensor(slot["x"])
            else:
                action_mask[i * 4 + j] = 0

    species = list(sample["ally"]["team"].keys())
    for i in range(6):
        if species[i] not in options["switches"]:
            action_mask[8 + i] = 0

    if choice["type"] == "move":
        i = int(options["tera"])
        j = [x["move"] for x in options["moves"]].index(choice["move"])
        target[i * 4 + j] = 1

    elif choice["type"] == "switch":
        i = species.index(choice["species"])
        target[8 + i] = 1

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
        action_mask=action_mask,
        target=target,
    )
