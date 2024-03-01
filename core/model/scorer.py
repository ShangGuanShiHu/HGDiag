from torch import nn
from .backone.FC import FullConnected



class Scorer(nn.Module):
    def __init__(self, in_dim, hiddens=[64, 128]):
        super(Scorer, self).__init__()
        self.net = FullConnected(in_dim, 1, hiddens)

    def forward(self, x):
        return self.net(x)


class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, hiddens=[64, 128]):
        super(Classifier, self).__init__()
        self.net = FullConnected(in_dim, out_dim, hiddens)

    def forward(self, x):
        return self.net(x)
