import torch
import numpy as np

from lookup import Config

def decode_states(c: Config, states, device):
    x = {}
    N = len(states)
    for _k, k, dtype, shape in [
        ("userType", "user_type", np.float32, (N, 7, 2, c.n_types)),
        ("userFeat", "user_feat", np.float32, (N, 7, c.user_feat_dim)),
        ("partyFeat", "party_feat", np.float32, (N, 2, c.party_feat_dim)),
        ("activeIdx", "active_idx", np.int32, (N, 2,)),
        ("moveChoiceIdx", "move_choice_idx", np.int32, (N, 4,)),
        ("moveMask", "move_mask", np.int32, (N, 4, 2)),
        ("switchMask", "switch_mask", np.int32, (N, 6,)),
    ]:
        a = b''.join([x[_k] for x in states])
        a = bytearray(a)
        a = np.frombuffer(a, dtype=dtype)
        a = a.reshape(shape)
        x[k] = torch.from_numpy(a).to(device)
    return x
