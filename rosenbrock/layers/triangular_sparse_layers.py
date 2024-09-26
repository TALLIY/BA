import torch
import torch.nn as nn

from rosenbrock.sparsity_masks.triangular_masks import (
    LowerTriangularMask,
    UpperTriangularMask,
)

# class UpperTraingularSparseLayer(nn.Module):
#     def __init__(self, layer_size):
#         super().__init__()
#         self.layer_size = layer_size
#         self.parameter_top = nn.Parameter(torch.rand([2]))
#         self.parameter_bottom = nn.Parameter(torch.rand([1]))

#     def forward(self, x: torch.Tensor):
#         return torch.matmul(
#             x,
#             torch.stack(
#                 [
#                     self.parameter_top,
#                     torch.cat(
#                         (
#                             self.parameter_bottom,
#                             torch.tensor([0]),
#                         ),
#                         dim=0,
#                     ),
#                 ],
#                 dim=0,
#             ).t(),
#         )


# class LowerTraingularSparseLayer(nn.Module):
#     def __init__(self, layer_size):
#         super().__init__()
#         self.parameter_top = nn.Parameter(torch.rand([1]))
#         self.parameter_bottom = nn.Parameter(torch.rand([2]))

#     def forward(self, x: torch.Tensor):
#         return torch.matmul(
#             x,
#             torch.stack(
#                 [
#                     torch.cat(
#                         (
#                             self.parameter_top,
#                             torch.tensor([0]),
#                         ),
#                         dim=0,
#                     ),
#                     self.parameter_bottom,
#                 ],
#                 dim=0,
#             ).t(),
#         )


class sparse_layer(nn.Linear):
    def __init__(self, dim, sparse_mask=None):
        super().__init__(dim, dim, bias=True)

        if sparse_mask is not None:
            self.sparse_mask = nn.Parameter(
                torch.Tensor(sparse_mask), requires_grad=False
            )

    def forward(self, x):
        if self.sparse_mask is not None:
            sparse_weight = self.weight * self.sparse_mask
            output = torch.matmul(x, sparse_weight.t())
        else:
            output = torch.matmul(x, self.weight.t())

        if self.bias is not None:
            output += self.bias

        return output


def UpperTraingularSparseLayer(dim):
    layer = sparse_layer(dim, sparse_mask=UpperTriangularMask(dim))
    return layer


def LowerTraingularSparseLayer(dim):
    layer = sparse_layer(dim, sparse_mask=LowerTriangularMask(dim))
    return layer
