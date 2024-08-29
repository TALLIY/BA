from typing import List

import torch


class ComputationalGrapBuilder:
    def __init__(self):
        self.jacobian_chain: List[torch.Tensor] = []
        self.eye = None

    def construct_graph(self, result: torch.Tensor) -> List[torch.Tensor]:
        self.eye = torch.eye(result.shape[0])
        for i in range(len(result)):
            recursion_depth = 0
            grad_fn = result[i].grad_fn
            derivative = self._pick_derivative(
                result[i].grad_fn(torch.tensor(1.0)), result[i].grad_fn
            )

            self._add_to_jacobian_chain(derivative, recursion_depth)
            self._traverse_graph(grad_fn, recursion_depth, curent_seed=derivative)

        print(len(self.jacobian_chain))
        print(self.jacobian_chain[9][9])
        print(self.jacobian_chain[98][10])
        return self.jacobian_chain

    def _traverse_graph(
        self,
        function: torch.Node | None,
        recursion_depth: int,
        curent_seed: torch.Tensor,
    ):
        recursion_depth += 1
        next_functions: torch.Node = function.next_functions
        for _, function in enumerate(next_functions):
            if (
                function[0] is None
                or "AccumulateGrad" in function[0].__class__.__name__
                or "TBackward" in function[0].__class__.__name__
            ):
                continue

            derivative = self._pick_derivative(function[0](curent_seed), function[0])
            self._add_to_jacobian_chain(derivative, recursion_depth)
            self._traverse_graph(function[0], recursion_depth, curent_seed)

    def _add_to_jacobian_chain(
        self,
        derivative: torch.Tensor,
        index: int,
    ):
        if derivative.shape[0] == 1:
            derivative_row = derivative
        else:
            derivative_row = derivative.unsqueeze(0)
        if len(self.jacobian_chain) <= index:
            self.jacobian_chain.append(derivative_row)
        else:
            self.jacobian_chain[index] = torch.cat(
                (self.jacobian_chain[index], derivative_row), dim=0
            )

    def _pick_derivative(self, derivatives, function):
        if isinstance(derivatives, tuple):
            if "AddmmBackward" in function.__class__.__name__:
                return derivatives[1]
            if "MvBackward" in function.__class__.__name__:
                return derivatives[1]
            if "ViewBackward" in function.__class__.__name__:
                return derivatives[1]
        else:
            return derivatives
