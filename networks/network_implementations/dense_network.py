import torch.nn as nn


class DenseNetwork(nn.Module):
    def __init__(self, layer_size):
        super(DenseNetwork, self).__init__()
        self.layer = nn.Linear(layer_size, layer_size)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.layer(x)
        out = self.relu(out)
        for _ in range(3):
            out = self.layer(out)
            out = self.relu(out)
        out = self.layer(out)

        return out
