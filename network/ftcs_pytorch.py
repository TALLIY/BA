import sys
from typing import List

import matplotlib.pyplot as plt
import torch

sys.setrecursionlimit(2000)


class FTCS:
    def __init__(self, length, n, alpha, min_T, max_T):
        # simulation parameters
        self.L = length
        self.n = n
        self.dx = length / (n - 1)
        self.alpha = alpha
        self.dt_max = self.dx**2 / (2 * alpha)
        self.dt = 0.9 * self.dt_max
        self.r = (alpha * self.dt) / (self.dx**2)

        self.min_T = min_T
        self.max_T = max_T

        # a vector
        self.T_vec = torch.rand(self.n) * (max_T - min_T) + min_T

        # applying initial conditions
        self.boundary_0 = torch.rand(1).item() * (max_T - min_T) + min_T
        self.boundary_1 = torch.rand(1).item() * (max_T - min_T) + min_T
        self.T_vec[0] = self.boundary_0
        self.T_vec[-1] = self.boundary_1

        self.matrix_chain: List[torch.Tensor] = []
        self.current_matrix_chain_index = 0
        self.number_of_iterations = 0

        # initialise the matrix
        self.A = torch.zeros((n, n))
        for j in range(n):
            try:
                if j > 0 or j < n:
                    self.A[j][j - 1] = self.r
                    self.A[j][j] = 1 - 2 * self.r
                    self.A[j][j + 1] = self.r
            except:  # noqa: E722
                continue

        self.A[0][-1] = 0
        self.A[0, 0] = 1
        self.A[0, 1] = 0
        self.A[-1, -2] = 0
        self.A[-1, -1] = 1

    def finite_difference(self, number_of_iterations, graph=False):
        self.number_of_iterations = number_of_iterations
        A = torch.tensor(self.A)
        print(A)
        T_0 = torch.tensor(self.T_vec, requires_grad=True)

        print("Temperature with initial conditions")
        print(T_0)

        # a vector values at each timestep
        timesteps = [T_0]

        hook_handlers = []
        i = 0
        while i < number_of_iterations:
            y = torch.matmul(A, timesteps[i])
            hook_handle = y.register_hook(self._hook_fn)
            timesteps.append(y)
            hook_handlers.append(hook_handle)
            i += 1

        self._generate_matrix_chain(timesteps[-1], timesteps[0])

        print("Temperature for final timestep: ", T_0)
        print("Number of iterations: ", len(timesteps))
        print("Total simulation time: ", len(timesteps) * self.dt)

        if graph:
            for j in range(1, len(timesteps)):
                if j % 100 == 0:
                    plt.plot(
                        torch.linspace(0, self.L, self.n).numpy(),
                        timesteps[j].detach().numpy(),
                        label="t = %.2f s" % (j * self.dt),
                    )

            plt.title("1D heat equation in a rod of length 1")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.grid(True)
            plt.xlabel("Length", fontsize=14)
            plt.ylabel("Temperature", fontsize=14)
            plt.tight_layout()
            plt.show()

        print("dim: ", len(timesteps[-1]))

    def _generate_matrix_chain(self, result: torch.Tensor, input: torch.Tensor):
        for i in range(len(result)):
            input.grad = None
            result[i].backward(retain_graph=True)

    def _hook_fn(self, grad: torch.Tensor):
        grad_row = grad.unsqueeze(0)
        index = self.current_matrix_chain_index
        if len(self.matrix_chain) < self.number_of_iterations:
            self.matrix_chain.append(grad_row)
        else:
            self.matrix_chain[index] = torch.cat(
                (self.matrix_chain[index], grad_row), dim=0
            )
            self._increment_matrix_chain_index()

    def _increment_matrix_chain_index(self):
        if self.current_matrix_chain_index + 1 >= self.number_of_iterations:
            self.current_matrix_chain_index = 0
        else:
            self.current_matrix_chain_index += 1

    # def _construct_computational_graph_recursive(self, result: torch.Tensor):
    #     matrix_chain: List[torch.Tensor] = []
    #     for i in range(len(result)):
    #         seed = torch.tensor(1.0)
    #         grad_fn = result[i].grad_fn
    #         derivatives = grad_fn(seed)
    #         recursion_depth = 0

    #         if isinstance(derivatives, tuple):
    #             for _, derivative in enumerate(derivatives):
    #                 if derivative is None:
    #                     continue

    #                 self._add_matrix_to_chain_recursive(
    #                     recursion_depth, derivative, matrix_chain
    #                 )
    #                 self._get_der_from_node_recursive(
    #                     derivative, grad_fn, matrix_chain, recursion_depth
    #                 )
    #         else:
    #             self._add_matrix_to_chain_recursive(
    #                 recursion_depth, derivatives, matrix_chain
    #             )
    #             self._get_der_from_node_recursive(
    #                 derivatives, grad_fn, matrix_chain, recursion_depth
    #             )
    #     print(matrix_chain[-1])

    # def _get_der_from_node_recursive(
    #     self,
    #     prev_derivative: Tuple[Union[torch.Tensor, Literal[0]], ...],
    #     grad_fn: torch.Node | None,
    #     matrix_chain: List[torch.Tensor],
    #     recursion_depth,
    # ):
    #     recursion_depth += 1
    #     next_functions = grad_fn.next_functions

    #     for _, function in enumerate(next_functions):
    #         if function[0] is None or prev_derivative is None:
    #             continue
    #         derivatives = function[0](prev_derivative)
    #         if isinstance(derivatives, tuple):
    #             for derivative in derivatives:
    #                 if derivative is None:
    #                     continue
    #                 self._add_matrix_to_chain_recursive(
    #                     recursion_depth, derivative, matrix_chain
    #                 )
    #                 self._get_der_from_node_recursive(
    #                     derivative, function[0], matrix_chain, recursion_depth
    #                 )
    #         else:
    #             self._add_matrix_to_chain_recursive(
    #                 recursion_depth, derivatives, matrix_chain
    #             )
    #             self._get_der_from_node_recursive(
    #                 derivatives, function[0], matrix_chain, recursion_depth
    #             )

    # def _add_matrix_to_chain_recursive(
    #     self,
    #     index: int,
    #     derivative: torch.Tensor,
    #     matrix_chain: List[torch.Tensor],
    # ):
    #     derivative_row = derivative.unsqueeze(0)
    #     if index >= len(matrix_chain):
    #         matrix_chain.append(derivative_row)
    #     else:
    #         matrix_chain[index] = torch.cat(
    #             (matrix_chain[index], derivative_row), dim=0
    #         )


sim = FTCS(length=1, n=101, alpha=0.01, min_T=0, max_T=100)

sim.finite_difference(number_of_iterations=1000, graph=False)
