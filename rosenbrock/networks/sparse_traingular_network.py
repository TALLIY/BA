import torch.nn as nn

from rosenbrock.layers.triangular_sparse_layers import (
    LowerTraingularSparseLayer,
    UpperTraingularSparseLayer,
)


class SpareTraingularNetwork(nn.Module):
    def __init__(self, layer_size):
        super(SpareTraingularNetwork, self).__init__()
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

        out = self.Upperlayer2(x)
        out = self.Lowerlayer2(out)
        out = self.activation(out)

        out = self.Upperlayer3(x)
        out = self.Lowerlayer3(out)
        out = self.activation(out)

        out = self.Upperlayer4(x)
        out = self.Lowerlayer4(out)
        out = self.activation(out)

        return out
