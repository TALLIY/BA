import torch
import torch.nn as nn


class UpperTraingularSparseLayer(nn.Module):
    def __init__(self, layer_size):
        super().__init__()
        self.layer_size = layer_size
        self.parameter_top = nn.Parameter(torch.rand([2]))
        self.parameter_bottom = nn.Parameter(torch.rand([1]))

    def forward(self, x: torch.Tensor):
        return torch.matmul(
            x,
            torch.stack(
                [
                    self.parameter_top,
                    torch.cat(
                        (
                            self.parameter_bottom,
                            torch.tensor([0]),
                        ),
                        dim=0,
                    ),
                ],
                dim=0,
            ).t(),
        )


class LowerTraingularSparseLayer(nn.Module):
    def __init__(self, layer_size):
        super().__init__()
        self.parameter_top = nn.Parameter(torch.rand([1]))
        self.parameter_bottom = nn.Parameter(torch.rand([2]))

    def forward(self, x: torch.Tensor):
        return torch.matmul(
            x,
            torch.stack(
                [
                    torch.cat(
                        (
                            self.parameter_top,
                            torch.tensor([0]),
                        ),
                        dim=0,
                    ),
                    self.parameter_bottom,
                ],
                dim=0,
            ).t(),
        )
