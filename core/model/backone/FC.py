from torch import nn


class FullConnected(nn.Module):
    def __init__(self, in_dim, out_dim, hiddens=[64]):
        super(FullConnected, self).__init__()

        self.net = nn.Sequential()
        in_dims = [in_dim] + hiddens[:-1].copy()
        out_dims = hiddens.copy()
        for net_in, net_out in zip(in_dims, out_dims):
            self.net.append(nn.Linear(net_in, net_out))
            self.net.append(nn.ReLU())
        self.net.append(nn.Linear(hiddens[-1], out_dim))

    def forward(self, x):
        return self.net(x)
