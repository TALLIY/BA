import torch


def f(x: torch.Tensor) -> torch.Tensor:
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    B = A.T + torch.eye(2)

    sigma = torch.nn.SiLU()

    return A @ sigma(B @ x)


x = torch.tensor([1.0, 2.0], requires_grad=True)

y = f(x)


Jacobian_Chain = []


def traverse_graph(grad_fn):
    # print(grad_fn)

    if hasattr(grad_fn, "_saved_self"):
        if grad_fn.__class__.__name__ == "MvBackward0":
            Jacobian_Chain.append(grad_fn._saved_self)

        else:
            partial = grad_fn(torch.tensor([1.0])).detach()

            Jacobian_Chain.append(torch.diag(partial))

    elif grad_fn.__class__.__name__ == "AccumulateGrad":
        pass

    else:
        raise NotImplementedError

    for next_grad_fn, _ in grad_fn.next_functions:
        if next_grad_fn is not None:
            traverse_graph(next_grad_fn)


# Traverse the graph and extract the Jacobian chain

traverse_graph(y.grad_fn)


print("Jacobian Chain:")

for J_local in Jacobian_Chain:
    print(J_local)


# Sanity check

print("\nSanity Check:")

J = torch.func.jacfwd(f)(x).detach()

print(J)


J_accumulated = torch.eye(2)

for J_local in Jacobian_Chain:
    J_accumulated = J_accumulated @ J_local

    print(J_accumulated)
