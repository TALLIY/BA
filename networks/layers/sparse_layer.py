import torch.nn as nn
import torch

class sparse_layer(nn.Linear):
    def __init__(self, input_size, output_size, sparse_mask=None):
        super(sparse_layer, self).__init__(input_size, output_size, bias=None)

        if sparse_mask is not None:
            self.sparse_mask = nn.Parameter(
                torch.Tensor(sparse_mask), requires_grad=False
            )
        else:
            self.sparse_mask = sparse_mask

    def forward(self, x):
        if self.sparse_mask is not None:
            sparse_weight = self.weight * self.sparse_mask
            return torch.matmul(x, sparse_weight.t())
        else:
            return torch.matmul(x, self.weight.t())