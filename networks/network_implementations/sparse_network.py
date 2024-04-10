import torch.nn as nn

from layers.triangular_layers import upper_traingular_layer, lower_traingular_layer

class SparseNetwork(nn.Module):
    def __init__(self, layer_size):
        super(SparseNetwork, self).__init__()
        self.upper_traingular = upper_traingular_layer(layer_size, layer_size)
        self.lower_traingular = lower_traingular_layer(layer_size, layer_size)
        self.relu = nn.ReLU()

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