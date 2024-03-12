import math

import torch
from torch import nn
from .backone.FC import FullConnected
from .backone.ResidualBlock import ResidualBlock
from torch.functional import F


# class Scorer(nn.Module):
#     def __init__(self, in_dim, hidden_dim=None, num_residual_blocks=3, feat_drop=0.3):
#         super(Scorer, self).__init__()
#         if hidden_dim==None:
#             hidden_dim = int(math.sqrt(in_dim))
#         self.net = FullConnected(in_dim, 1, [hidden_dim for i in range(num_residual_blocks)])
#
#     def forward(self, x):
#         return self.net(x)


class Scorer(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, num_residual_blocks=3, feat_drop=0.3):
        super(Scorer, self).__init__()
        if hidden_dim==None:
            hidden_dim = int(math.sqrt(in_dim))
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, hidden_dim, feat_drop=feat_drop) for _ in range(num_residual_blocks)])
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc_in(x))
        for block in self.residual_blocks:
            x = block(x)
        x = self.fc_out(x)
        # x = torch.clamp(x, min=0.00001,max=5)
        # x = torch.clamp(x, min=1e-7, max=50)
        # x = torch.clamp(x, min=0, max=500)
        # x = torch.clamp(x, max=5)
        return x

    def clip_grad_norm(self, max_norm=100):
        torch.nn.utils.clip_grad_norm(self.fc_out.parameters(), max_norm=max_norm)


class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, hiddens=[64, 128]):
        super(Classifier, self).__init__()
        self.net = FullConnected(in_dim, out_dim, hiddens)

    def forward(self, x):
        return self.net(x)
