import copy
import csv

import matplotlib.pyplot as plt
import numpy as np


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
        self.T_vec = np.random.uniform(min_T, max_T, size=self.n)

        # applying initial conditions
        self.boundary_0 = np.random.uniform(min_T, max_T)
        self.boundary_1 = np.random.uniform(min_T, max_T)
        self.T_vec[0] = self.boundary_0
        self.T_vec[-1] = self.boundary_1

        # initialise the matrix
        self.A = np.zeros(shape=(n, n))
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

    def generate_function(self, dim, path):
        A_dim = np.linalg.matrix_power(self.A, dim)
        np.save(
            f"{path}/A_{dim}_bound({round(self.boundary_0,0)}-{round(self.boundary_1,0)})_maxmin({round(self.min_T,0)}-{round(self.max_T,0)}).npy",
            A_dim,
        )

    def finite_difference(self, number_of_iterations, graph=False):
        n = self.n
        T_vec = self.T_vec
        dt = self.dt

        print("Temperature with initial conditions")
        print(T_vec)

        # a vector values at each timestep
        timesteps = [T_vec]
        i = 0
        while i < number_of_iterations:
            X = np.dot(self.A, T_vec)
            T_vec = copy.deepcopy(X)
            timesteps.append(T_vec)
            i += 1

        print("Temperature for final timestep: ", T_vec)
        print("Number of iterations: ", len(timesteps))
        print("Total simulation time: ", len(timesteps) * dt)

        if graph:
            for j in range(1, len(timesteps)):
                if j % 1000 == 0:
                    plt.plot(
                        list(np.linspace(0, self.L, n)),
                        timesteps[j],
                        label="t = %.2f s" % (j * dt),
                    )
            plt.title("1D heat equation in a rod of length 1")
            plt.legend(bbox_to_anchor=[1, 1])
            plt.grid(True)
            plt.xlabel("Length", fontsize=14)
            plt.ylabel("Temperature", fontsize=14)
            plt.tight_layout()
            plt.show()

        print("dim: ", len(timesteps[-1]))

        return timesteps

    def generate_data(self, number_of_iterations, number_of_datapoints, path):
        m = number_of_datapoints
        for i in range(0, m):
            self._genrate_data_helper(number_of_iterations, path)
            print(f"run [{i + 1}/{m}] done")

    def _genrate_data_helper(self, number_of_iterations, path):
        T_vec = np.random.uniform(self.min_T, self.max_T, size=self.n)
        boundary_0 = np.random.uniform(self.min_T, self.max_T)
        boundary_1 = np.random.uniform(self.min_T, self.max_T)
        T_vec[0] = boundary_0
        T_vec[-1] = boundary_1
        timesteps = [T_vec]

        i = 0
        while i < number_of_iterations:
            X = np.dot(self.A, T_vec)
            T_vec = copy.deepcopy(X)
            timesteps.append(T_vec)
            i += 1

        print(len(timesteps[-1]))
        self._save_data_to_csv(timesteps, path)

    def _save_data_to_csv(self, timesteps, path):
        with open(path, "a", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(timesteps[0])
            csv_writer.writerow(timesteps[-1])


sim = FTCS(length=1, n=101, alpha=0.01, min_T=0, max_T=100)

sim.generate_function(10000, "../shared/generated_functions")

# sim.finite_difference(number_of_iterations=10000, graph=True)

# sim.generate_data(
#     number_of_iterations=10000,
#     number_of_datapoints=1000,
#     path="../shared/datasets/ftcs/training_n_101.csv",
# )
