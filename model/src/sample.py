import torch


def one_hot_types(lookup, types):
    x = torch.zeros(lookup.dim["types"])
    x[[lookup.get_type(t) for t in types]] = 1


def vectorize_sample(idx, sample):
    battle = sample["battle"]
    options = sample["options"]
    choice = sample["choice"]

    move_idx = torch.zeros(2, 6, 15)
    move_x = torch.zeros(2, 6, 15, 2)
    move_pool_n = torch.zeros(2, 6)
    move_mask = torch.ones(2, 6, 5)
    move_max_mask = torch.zeros(2, 6, 5)

    ability_x = torch.zeros(2, 6, 3)
    ability_n = torch.zeros(2, 6)
    ability_mask = torch.ones(2, 6, 3)

    item_x = torch.zeros(2, 6, 4)
    item_n = torch.zeros(2, 6)
    item_mask = torch.ones(2, 6, 4)

    user_x = torch.zeros(2, 6)
    user_mask = torch.ones(2, 6)

    side_x = torch.zeros(2, 100)
    active_idx = torch.zeros(2)

    battle_x = torch.tensor(battle["x"])

    move_option_idx = torch.zeros(4)
    action_mask = torch.zeros(14)

    choice_x = torch.zeros(14)

    sides = [sample["ally"], sample["foe"]]
    for i in range(2):
        side = sides[i]
        team = side["team"]
        species = list(team.keys())

        side_x[i] = side["x"]
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
                        user["x"],
                        one_hot_types(idx, types),
                        one_hot_types(idx, tera_types) / len(tera_types),
                    ]
                )

                for k in range(4):
                    if k < len(move_set):
                        slot = move_set[k]
                        move_idx[i][j][k] = idx.moves[slot["move"]]
                        move_x[i][j][k] = slot["x"]
                    else:
                        move_max_mask[i][j][k] = float("-inf")

                move_pool_n[i][j] = min(len(move_pool), 6)
                for k in range(6):
                    if k < len(move_pool):
                        slot = move_pool[k]
                        move_idx[i][j][4 + k] = idx.moves[slot["move"]]
                        move_x[i][j][4 + k] = slot["x"]
                    else:
                        move_mask[i][j][4 + k] = 0

                for k, ref in enumerate(
                    ["disabled", "choice", "encore", "locked", "lastMove"]
                ):
                    if ref in user:
                        slot = user[ref]
                        move_idx[i][j][10 + k] = idx.moves[slot["move"]]
                        move_x[i][j][10 + k] = slot["x"]
                    else:
                        move_mask[i][j][10 + k] = 0

                ability_n[i][j] = min(len(abilities), 3)
                for k in range(3):
                    if k < len(abilities):
                        ability_x[i][j][k] = idx.abilities[abilities[k]]
                    else:
                        ability_mask[i][j][k] = 0

                item_n[i][j] = min(len(items), 3)
                for k in range(3):
                    if k < len(items):
                        item_x[i][j][k] = idx.items[items[k]]
                    else:
                        item_mask[i][j][k] = 0

                for k, ref in enumerate(["lastBerry"]):
                    if ref in user:
                        item_x[i][j][3 + k] = idx.items[user[ref]]
                    else:
                        item_mask[i][j][3 + k] = 0
            else:
                user_mask[i][j] = float("-inf")

    for i in range(2):
        for j in range(4):
            if not ((options["canTera"] or i == 0) and j < len(options["moves"])):
                action_mask[i * 4 + j] = float("-inf")

    species = list(sample["ally"]["team"].keys())
    for i in range(6):
        if species[i] not in options["switches"]:
            action_mask[8 + i] = float("-inf")

    if choice["type"] == "move":
        i = int(options["tera"])
        j = [x["move"] for x in options["moves"]].index(choice["move"])
        action_mask[i * 4 + j] = 1
    elif choice["type"] == "switch":
        i = species.index(choice["species"])
        choice_x[8 + i] = 1

    return dict(
        move_base_x=move_idx,
        move_slot_x=move_x,
        move_pool_n=move_pool_n,
        move_mask=move_mask,
        move_max_mask=move_max_mask,
        ability_x=ability_x,
        ability_n=ability_n,
        ability_mask=ability_mask,
        item_x=item_x,
        item_n=item_n,
        item_mask=item_mask,
        user_x=user_x,
        user_mask=user_mask,
        side_x=side_x,
        active_idx=active_idx,
        battle_x=battle_x,
        move_option_idx=move_option_idx,
        action_mask=action_mask,
        choice_x=choice_x,
    )
