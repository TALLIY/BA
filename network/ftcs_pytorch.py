import sys
from typing import List

import matplotlib.pyplot as plt
import torch
from computational_graph_builder import ComputationalGrapBuilder

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

        self.cgb = ComputationalGrapBuilder()

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

        T_0 = torch.tensor(self.T_vec, requires_grad=True)

        print("Temperature with initial conditions")
        print(T_0)

        # a vector values at each timestep
        timesteps = [T_0]

        i = 0
        while i < number_of_iterations:
            y = torch.matmul(A, timesteps[i])
            timesteps.append(y)
            i += 1

        self.cgb.construct_graph(timesteps[-1])

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


sim = FTCS(length=1, n=101, alpha=0.01, min_T=0, max_T=100)

sim.finite_difference(number_of_iterations=1000, graph=False)
