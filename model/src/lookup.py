import torch

N_TYPES = 20


def get_embeddings(db, device):
    items = {
        f["name"]: torch.tensor(f["desc"]["openai"], device=device)
        for f in db.items.find()
    }
    abilities = {
        f["name"]: torch.tensor(f["desc"]["openai"], device=device)
        for f in db.abilities.find()
    }
    moves = {
        f["name"]: torch.tensor(f["x"] + f["desc"]["openai"], device=device)
        for f in db.moves.find()
    }

    return dict(items=items, abilities=abilities, moves=moves, types=types)
