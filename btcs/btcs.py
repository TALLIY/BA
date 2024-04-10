import copy
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


class BTCS:
    def __init__(self, length, delta_x, alpha, delta_t, min_T, max_T):
        # simulation parameters
        self.L = length
        self.dx = delta_x
        self.alpha = alpha
        self.dt = delta_t
        self.n = int(length / delta_x - 1)
        self.min_T = min_T
        self.max_T = max_T

        # a vector
        self.a_vec = np.random.uniform(min_T, max_T, size=self.n)

        # applying initial conditions
        self.boundary_0 = np.random.uniform(min_T, max_T)
        self.boundary_1 = np.random.uniform(min_T, max_T)
        self.a_vec[0] = self.boundary_0
        self.a_vec[-1] = self.boundary_1

        # initialise the matrix
        self.A = np.zeros(shape=(self.n, self.n))
        for j in range(self.n):
            try:
                self.A[j][j - 1] = -self.alpha / (self.dx**2)
                self.A[j][j] = 1 / self.dt + 2 * alpha / (self.dx**2)
                self.A[j][j + 1] = -self.alpha / (self.dx**2)
            except:  # noqa: E722
                continue
        self.A[0][-1] = 0  # prevent the appearance of a rouge zero

        # applying drichlet boundary conditions
        self.A[0, 0] = 1
        self.A[0, 1] = 0
        self.A[-1, -2] = 0
        self.A[-1, -1] = 1

    def finite_difference(self, number_of_iterations, graph=False):
        dt = self.dt
        n = self.n
        a_vec = self.a_vec
        boundary_0 = self.boundary_0
        boundary_1 = self.boundary_1
        A = self.A

        print("Temperature with initial conditions")
        print(a_vec)

        # a vector values at each timestep
        timesteps = [a_vec]
        i = 0
        while i < number_of_iterations:
            b = (1 / dt) * copy.deepcopy(a_vec)
            b[0] = boundary_0
            b[-1] = boundary_1

            X = np.linalg.solve(A, b)
            a_vec = copy.deepcopy(X)
            timesteps.append(a_vec)
            i += 1

        print("Temperature for final timestep: ", a_vec)
        print("Number of iterations: ", len(timesteps))
        print("Total simulation time: ", len(timesteps) * dt)

        if graph:
            for j in range(1, len(timesteps)):
                if j % 100 == 0:
                    plt.plot(
                        list(np.linspace(0, 1, n)),
                        timesteps[j],
                        label="t = %.2f s" % (j * dt),
                    )
            plt.title("1D heat equation in a rod of length 1")
            plt.legend(bbox_to_anchor=[1, 1])
            plt.grid(True)
            plt.xlabel("Length", fontsize=14)
            plt.ylabel("Temperature", fontsize=14)
            plt.show()

        return timesteps

    def infer_final_step_value(
        self, model, state_dict_path, denormalisation_params=None
    ):
        model.load_state_dict(torch.load(state_dict_path))
        model.eval()
        min, max = denormalisation_params
        f = np.random.uniform(self.min_T, self.max_T, size=self.n)
        surr_f = (
            model(torch.tensor(f.astype(np.float32))).detach() * (max - min)
        ) + min

        return surr_f

    def generate_data(self, number_of_iterations, datapoints):
        m = datapoints
        for i in range(0, m):
            self._genrate_data_helper(number_of_iterations)
            print(f"run [{i + 1}/{m}] done")

    def _genrate_data_helper(self, number_of_iterations):
        dt = self.dt
        a_vec = self.a_vec
        boundary_0 = self.boundary_0
        boundary_1 = self.boundary_1
        A = self.A

        print("Temperature with initial conditions")
        print(a_vec)

        # a vector values at each timestep
        timesteps = [a_vec]
        i = 0
        while i < number_of_iterations:
            b = (1 / dt) * copy.deepcopy(a_vec)
            b[0] = boundary_0
            b[-1] = boundary_1

            X = np.linalg.solve(A, b)
            a_vec = copy.deepcopy(X)
            timesteps.append(a_vec)
            i += 1
        self._save_data_to_csv(timesteps, "training.csv")

    def _save_data_to_csv(self, timesteps, filename):
        if self._file_exists("../datasets", filename):
            with open(filename, "a", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(timesteps[0])
                csv_writer.writerow(timesteps[-1])
        else:
            last_char = filename.split(".")[0][-1]
            if last_char.isdigit():
                with open(
                    f"{filename.split(".")[0]}{int(last_char)+1}.{filename.split(".")[1]}",
                    "a",
                    newline="",
                ) as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(timesteps[0])
                    csv_writer.writerow(timesteps[-1])
            else:
                with open(
                    f"{filename.split(".")[0]}2.{filename.split(".")[1]}",
                    "a",
                    newline="",
                ) as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(timesteps[0])
                    csv_writer.writerow(timesteps[-1])

    def _file_exists(directory, filename):
        filepath = os.path.join(directory, filename)
        return os.path.exists(filepath) and os.path.isfile(filepath)
