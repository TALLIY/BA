from typing import List

import torch


class ComputationalGrapBuilder:
    def __init__(self, result: torch.Tensor):
        self.result: torch.Tensor = result
        self.matrix_chain: List[torch.Tensor] = []

    def construct_graph(self):
        result = self.result
        for i in range(len(result)):
            recursion_depth = 0
            grad_fn = result[i].grad_fn
            derivatives = result[i].grad_fn(torch.tensor(1.0))
            if isinstance(derivatives, tuple):
                for index, derivative in enumerate(derivatives):
                    if derivative is None:
                        continue
                    if index > 1:
                        raise Exception("More indices than expected")

                    self._add_to_matrix_chain(derivative, recursion_depth)
                    self._traverse_graph(
                        grad_fn, recursion_depth, curent_seed=derivative
                    )
            else:
                self._add_to_matrix_chain(derivatives, recursion_depth)
                self._traverse_graph(grad_fn, recursion_depth, curent_seed=derivatives)

        print(len(self.matrix_chain))
        for matrix in self.matrix_chain:
            print(matrix)

    def _traverse_graph(self, function, recursion_depth, curent_seed):
        recursion_depth += 1
        next_functions = function.next_functions
        for _, function in enumerate(next_functions):
            if function[0] is None:
                continue
            derivatives = function[0](curent_seed)
            if isinstance(derivatives, tuple):
                for index, derivative in enumerate(derivatives):
                    if derivative is None:
                        continue
                    if index > 1:
                        raise Exception("More indices than expected")

                    self._add_to_matrix_chain(derivative, recursion_depth)
                    self._traverse_graph(function[0], recursion_depth, curent_seed)

            else:
                self._add_to_matrix_chain(derivatives, recursion_depth)
                self._traverse_graph(function[0], recursion_depth, curent_seed)

    def _add_to_matrix_chain(
        self,
        derivative: torch.Tensor,
        index: int,
    ):
        derivative_row = derivative.unsqueeze(0)
        print(index)
        print(len(self.matrix_chain))
        if len(self.matrix_chain) <= index:
            self.matrix_chain.append(derivative_row)
        else:
            self.matrix_chain[index] = torch.cat(
                (self.matrix_chain[index], derivative_row), dim=0
            )
