import torch.nn as nn
from triangular_layers import UpperTraingularSparseLayer

from rosenbrock.layers.triangular_sparse_layers import LowerTraingularSparseLayer


class dense_network(nn.Module):
    def __init__(self, layer_size):
        super(dense_network, self).__init__()
        self.layer1 = nn.Linear(layer_size, layer_size, bias=False)
        self.layer2 = nn.Linear(layer_size, layer_size, bias=False)
        self.layer3 = nn.Linear(layer_size, layer_size, bias=False)
        self.layer4 = nn.Linear(layer_size, layer_size, bias=False)
        self.leakyRelu = nn.LeakyReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.leakyRelu(out)
        out = self.layer2(out)
        out = self.leakyRelu(out)
        out = self.layer3(out)
        out = self.leakyRelu(out)
        out = self.layer4(out)
        return out


class sparse_network(nn.Module):
    def __init__(self, layer_size):
        super().__init__()
        self.Upperlayer1 = UpperTraingularSparseLayer(layer_size)
        self.Lowerlayer1 = LowerTraingularSparseLayer(layer_size)

        self.Upperlayer2 = UpperTraingularSparseLayer(layer_size)
        self.Lowerlayer2 = LowerTraingularSparseLayer(layer_size)

        self.Upperlayer3 = UpperTraingularSparseLayer(layer_size)
        self.Lowerlayer3 = LowerTraingularSparseLayer(layer_size)

        self.Upperlayer4 = UpperTraingularSparseLayer(layer_size)
        self.Lowerlayer4 = LowerTraingularSparseLayer(layer_size)

        self.activation = nn.Softplus()

    def forward(self, x):
        out = self.Upperlayer1(x)
        out = self.Lowerlayer1(out)
        out = self.activation(out)

        out = self.Upperlayer2(out)
        out = self.Lowerlayer2(out)
        out = self.activation(out)

        out = self.Upperlayer3(out)
        out = self.Lowerlayer3(out)

        return out
