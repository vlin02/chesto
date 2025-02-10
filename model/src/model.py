from torch import nn

MAX_ITEM_NUM = 2500



class UserNet(nn.Module):
    def __init__(self):
        self.
        
    def forward(user):
        
class Net(nn.Module):
    def __init__(self, item_embed, ability_embed):
        super().__init__()
        self.fc_item = nn.Linear(256, 128)
        self.fc_ability = nn.Linear(256, 128)
        self.fc_move = nn.Linear(1000, 256)

        self.fc_user = nn.Linear(100, 512)

    def forward(self, obs):
        obs["f"]


