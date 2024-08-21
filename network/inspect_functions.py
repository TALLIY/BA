import numpy as np
import torch
from torch.func import jacfwd
from torchviz import make_dot

A_dim = np.load(
    "../shared/generated_functions/A_10000_bound(20.0-73.0)_maxmin(0-100).npy"
)

A = torch.tensor(np.array(A_dim))


def func(x: torch.Tensor):
    y = torch.cos(x) ** 2
    z = torch.log(y)
    w = torch.sin(z)
    return w


T = np.random.uniform(0, 100, size=101)
T[0] = 20
T[-1] = 73

x = torch.tensor(T, requires_grad=True)


jacobian = jacfwd(func)(x)

dot = make_dot(jacobian, params={"x": x})
dot.render("jacobian_graph", format="png", cleanup=True)


def traverse_graph(var, nodes=set(), edges=set(), node_info={}, edge_info=[]):
    print("dir(var)")
    print(var.next_functions[0])
    if var not in nodes:
        nodes.add(var)
        if hasattr(var, "variable"):
            u = var.variable
            node_info[var] = {
                "type": type(var).__name__,
                "shape": u.shape,
                "value": u.detach().cpu().numpy() if u.is_cuda else u.detach().numpy(),
            }
        else:
            node_info[var] = {"type": type(var).__name__, "shape": None, "value": None}

        if hasattr(var, "next_functions"):
            for u in var.next_functions:
                if u[0] is not None:
                    edges.add((u[0], var))
                    edge_info.append(
                        {
                            "from": u[0],
                            "to": var,
                            "from_type": type(u[0]).__name__,
                            "to_type": type(var).__name__,
                        }
                    )
                    traverse_graph(u[0], nodes, edges, node_info, edge_info)
        if hasattr(var, "saved_tensors"):
            for t in var.saved_tensors:
                edges.add((t, var))
                edge_info.append(
                    {
                        "from": t,
                        "to": var,
                        "from_type": type(t).__name__,
                        "to_type": type(var).__name__,
                    }
                )
                traverse_graph(t, nodes, edges, node_info, edge_info)
    return nodes, edges, node_info, edge_info


nodes, edges, node_info, edge_info = traverse_graph(jacobian.grad_fn)

print("Nodes:")
for node, info in node_info.items():
    print(f"{info['type']} (shape: {info['shape']}, value: {info['value']})")

print("\nEdges:")
for edge in edge_info:
    print(f"{edge['from_type']} -> {edge['to_type']}")
