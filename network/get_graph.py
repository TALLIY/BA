import numpy as np
import torch
import torch.fx as fx

fx.wrap("len")

A_dim = np.load(
    "../shared/generated_functions/A_10000_bound(20.0-73.0)_maxmin(0-100).npy"
)

A = torch.tensor(np.array(A_dim))


def func(x):
    return torch.matmul(A, x)


T = np.random.uniform(0, 100, size=101)
T[0] = 20
T[-1] = 73
x = torch.tensor(T, requires_grad=True)


class FuncModule(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        res = func(x)
        res.backward()
        return x.grad


traced_module = fx.symbolic_trace(FuncModule())
print(traced_module.graph)
traced_module.graph.print_tabular()
