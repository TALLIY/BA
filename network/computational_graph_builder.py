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
            self._traverse_graph(function[0], recursion_depth, derivative)

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
            elif "MvBackward" in function.__class__.__name__:
                return derivatives[1]
            elif "ViewBackward" in function.__class__.__name__:
                return derivatives[1]
            elif "SqueezeBackward" in function.__class__.__name__:
                return derivatives[0]
            elif "LeakyReluBackward" in function.__class__.__name__:
                return derivatives[0]
            elif "UnsqueezeBackward" in function.__class__.__name__:
                return derivatives[0]
            elif "MmBackward" in function.__class__.__name__:
                return derivatives[0]
            elif "MulBackward" in function.__class__.__name__:
                return derivatives[0]
            elif "AddBackward" in function.__class__.__name__:
                return derivatives[0]
            else:
                for index, derivative in enumerate(derivatives):
                    if index > 1:
                        raise Exception("More indices than expected")
                    if derivative is None:
                        continue
                    return derivative
        else:
            return derivatives
