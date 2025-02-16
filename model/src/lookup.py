import torch

N_TYPES = 20

def get_lookup(db, device):
    item_embed = torch.zeros(256, 128)
    ability_embed = torch.zeros(512, 128)
    move_embed = torch.zeros(1024, 228 + 256)

    item_idx = {}
    ability_idx = {}
    move_idx = {}
    type_idx = {}

    for item in db.items.find():
        item_embed[item["i"]] = torch.tensor(item["desc"]["openai"], device=device)
        item_idx[item["name"]] = item["i"]

    for ability in db.abilities.find():
        ability_embed[ability["i"]] = torch.tensor(
            ability["desc"]["openai"], device=device
        )
        ability_idx[ability["name"]] = ability["i"]

    for move in db.moves.find():
        move_embed[move["i"]] = torch.tensor(
            move["x"] + move["desc"]["openai"], device=device
        )
        move_idx[move["name"]] = move["i"]

    for type in db.types.find():
        type_idx[type["name"]] = type["i"]

    return dict(
        item_embed=item_embed,
        ability_embed=ability_embed,
        move_embed=move_embed,
        item_idx=item_idx,
        move_idx=move_idx,
        ability_idx=ability_idx,
        type_idx=type_idx,
    )
