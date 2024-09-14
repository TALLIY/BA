import torch
import torch.nn as nn


class UpperTraingularSparseLayer(nn.Module):
    def __init__(self, layer_size):
        super().__init__()
        self.layer_size = layer_size
        self.parameters_list = nn.ParameterList(
            [nn.Parameter(torch.rand(layer_size - i)) for i in range(layer_size)]
        )

    def forward(self, x: torch.Tensor):
        y = torch.zeros(self.layer_size, device=x.device, dtype=x.dtype)
        for i in range(self.layer_size):
            len_vector = len(self.parameters_list[i])
            for j in range(len_vector):
                y[i] += self.parameters_list[i][j] * x[self.layer_size - len_vector + j]
        return y


class LowerTraingularSparseLayer(nn.Module):
    def __init__(self, layer_size):
        super().__init__()
        self.layer_size = layer_size
        self.parameters_list = nn.ParameterList(
            [
                nn.Parameter(torch.rand(layer_size - i))
                for i in reversed(range(layer_size))
            ]
        )

    def forward(self, x: torch.Tensor):
        y = torch.zeros(self.layer_size, device=x.device, dtype=x.dtype)
        for i in range(self.layer_size):
            len_vector = len(self.parameters_list[i])
            for j in range(len_vector):
                y[i] += self.parameters_list[i][j] * x[j]

        return y
