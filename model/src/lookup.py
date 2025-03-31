from dataclasses import dataclass


USER_FEAT_DIM = 52
MOVE_FEAT_DIM = 30
PARTY_FEAT_DIM = 10
TYPE_COUNT = 20
HIDDEN_DIM = 256


@dataclass
class Config:
    n_types: int = TYPE_COUNT

    user_feat_dim: int = USER_FEAT_DIM
    move_feat_dim: int = MOVE_FEAT_DIM
    party_feat_dim: int = PARTY_FEAT_DIM

    hidden_dim: int = HIDDEN_DIM

    
