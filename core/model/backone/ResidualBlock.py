from torch import nn
from torch.functional import F


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        residual = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x += residual  # 残差连接
        x = F.relu(x)
        return x