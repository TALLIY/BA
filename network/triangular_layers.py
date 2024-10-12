import torch
import torch.nn as nn

from network.connectivity_masks import LowerTriangularMask, UpperTriangularMask


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
