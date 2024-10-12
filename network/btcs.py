import copy
import csv
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from computational_graph_builder import ComputationalGrapBuilder


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
                if j % 10 == 0:
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
            plt.tight_layout()
            plt.show()

        print(len(timesteps[-1]))

        return timesteps

    def infer_final_step_value(
        self,
        model,
        state_dict_path,
        denormalisation_params_path=None,
        compare=False,
        number_of_iterations=1000,
    ):
        model.load_state_dict(torch.load(state_dict_path))
        model.eval()

        checkpoint = torch.load(state_dict_path)
        model.load_state_dict(checkpoint)

        if denormalisation_params_path is not None:
            with open(denormalisation_params_path, "rb") as f:
                denormalisation_params = pickle.load(f)
            min, max = denormalisation_params["min"], denormalisation_params["max"]
            f = np.random.uniform(self.min_T, self.max_T, size=self.n)
            f[0] = self.boundary_0
            f[-1] = self.boundary_1
            input = torch.tensor(f.astype(np.float32), requires_grad=True)
            surr_f = (model(input) * (max - min)) + min

            cdp = ComputationalGrapBuilder()

            chain = cdp.construct_graph(surr_f)

            for matrix in chain:
                print(matrix)
        else:
            f = np.random.uniform(self.min_T, self.max_T, size=self.n)
            surr_f = model(torch.tensor(f.astype(np.float32))).detach()

        if compare:
            f = self.finite_difference(number_of_iterations)[-1]
            plt.plot(
                list(np.linspace(0, 1, self.n)),
                f,
                label="f sim",
            )
            plt.plot(
                list(np.linspace(0, 1, self.n)),
                surr_f.detach().numpy(),
                label="f inferred",
            )
            plt.title("1D heat equation in a rod of length 1")
            plt.legend(bbox_to_anchor=[1, 1])
            plt.grid(True)
            plt.xlabel("Length", fontsize=14)
            plt.ylabel("Temperature", fontsize=14)
            plt.tight_layout()
            plt.show()

        return surr_f

    def generate_data(self, number_of_iterations, number_of_datapoints, path):
        m = number_of_datapoints
        for i in range(0, m):
            self._genrate_data_helper(number_of_iterations, path)
            print(f"run [{i + 1}/{m}] done")

    def _genrate_data_helper(self, number_of_iterations, path):
        dt = self.dt
        a_vec = np.random.uniform(self.min_T, self.max_T, size=self.n)
        boundary_0 = np.random.uniform(self.min_T, self.max_T)
        boundary_1 = np.random.uniform(self.min_T, self.max_T)
        a_vec[0] = boundary_0
        a_vec[-1] = boundary_1
        A = np.zeros(shape=(self.n, self.n))
        for j in range(self.n):
            try:
                A[j][j - 1] = -self.alpha / (self.dx**2)
                A[j][j] = 1 / self.dt + 2 * self.alpha / (self.dx**2)
                A[j][j + 1] = -self.alpha / (self.dx**2)
            except:  # noqa: E722
                continue

        A[0][-1] = 0
        A[0, 0] = 1
        A[0, 1] = 0
        A[-1, -2] = 0
        A[-1, -1] = 1

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

        # print(len(timesteps[-1]))
        self._save_data_to_csv(timesteps, path)

    def _save_data_to_csv(self, timesteps, path):
        with open(path, "a", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(timesteps[0])
            csv_writer.writerow(timesteps[-1])


sim = BTCS(length=1, delta_x=0.005, alpha=0.01, delta_t=0.01, min_T=0, max_T=1000)

# sim.finite_difference(number_of_iterations=1000, graph=False)

# load_dotenv()
# train_dense_network = True
# if os.getenv("TRAIN_DENSE_NETWORK") == "0":
#     train_dense_network = False

# print(train_dense_network)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# sim = BTCS(length=1, delta_x=0.005, alpha=0.01, delta_t=0.01, min_T=0, max_T=1000)

# sim.finite_difference(number_of_iterations=100, graph=True)

# if train_dense_network:
#     model = dense_network(199).to(device)
# else:
#     model = sparse_network(199).to(device)

# sim.infer_final_step_value(
#     model=model,
#     state_dict_path="../shared/weights/model_weights_for_layer_size_199_for_dense_False.pth",
#     denormalisation_params_path="../shared/parameters/min_max_scaling_params_for_layer_size_199_for_dense_False.pkl",
#     compare=True,
#     number_of_iterations=100,
# )


# sim = BTCS(length=1, delta_x=0.002, alpha=0.01, delta_t=0.01, min_T=0, max_T=1000)

# sim.generate_data(
#     number_of_iterations=100,
#     number_of_datapoints=80000,
#     path="../shared/datasets/training_n_499.csv",
# )
