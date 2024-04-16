import torch.nn as nn
from triangular_layers import lower_traingular_layer, upper_traingular_layer


class dense_network(nn.Module):
    def __init__(self, layer_size):
        super(dense_network, self).__init__()
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


class sparse_network(nn.Module):
    def __init__(self, layer_size):
        super(sparse_network, self).__init__()
        self.upper_traingular = upper_traingular_layer(layer_size, layer_size)
        self.lower_traingular = lower_traingular_layer(layer_size, layer_size)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.upper_traingular(x)
        out = self.relu(out)
        for _ in range(1):
            out = self.lower_traingular(out)
            out = self.relu(out)
            out = self.upper_traingular(x)
            out = self.relu(out)
        out = self.lower_traingular(out)

        return out
