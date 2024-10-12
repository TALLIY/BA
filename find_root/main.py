import pickle
import sys
import warnings

import numpy as np
import torch
from torch.func import jacfwd

from network.computational_graph_builder import ComputationalGrapBuilder
from rosenbrock.networks.sparse_traingular_network import SpareTraingularNetwork

sys.setrecursionlimit(2000)


warnings.filterwarnings("ignore")


def min_max_denormalise(
    input: torch.Tensor, min: torch.Tensor | float, max: torch.Tensor | float
):
    return (input * (max - min)) + min


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpareTraingularNetwork(2).to(device)
model.double()
checkpoint = torch.load(
    "/Users/talli/BA/rosenbrock/saved_weights/rosenbrock_trained_model_weights_sparse_prod.pth"
)

model.load_state_dict(checkpoint)
model.eval()
with open(
    "/Users/talli/BA/rosenbrock/saved_params/rosenbrock_sampled_data_normalisation_params_2.pkl",
    "rb",
) as f:
    denormalisation_params = pickle.load(f)
minimum, maximum = denormalisation_params["min"], denormalisation_params["max"]


length = 1
n = 30
alpha = 20

dx = length / (n - 1)
dt_max = dx**2 / (2 * alpha)
# dt_max_cfl = dx / alpha
dt = 0.2 * dt_max

r = (alpha * dt) / (dx**2)

print("r: ", r)

A = torch.zeros((n, n)).double()
for j in range(n):
    try:
        if j > 0 or j < n:
            A[j][j - 1] = r
            A[j][j] = 1 - 2 * r
            A[j][j + 1] = r
    except:  # noqa: E722
        continue

A[0][-1] = 0
A[0, 0] = 1
A[0, 1] = 0
A[-1, -2] = 0
A[-1, -1] = 1

print(A.dtype)
number_of_iterations = 8

print(
    "svals: ", torch.linalg.svdvals(torch.linalg.matrix_power(A, number_of_iterations))
)
print(
    "condition: ", torch.linalg.cond(torch.linalg.matrix_power(A, number_of_iterations))
)


def heat_equation_pytorch(input):
    timesteps = [input]

    i = 0
    while i < number_of_iterations:
        y = torch.tensor(A) @ timesteps[i]
        timesteps.append(y.clone())
        i += 1

    # if True:
    #     print("Temperature for final timestep: ", timesteps[-1])
    #     print("Number of iterations: ", len(timesteps) - 1)
    #     print("Total simulation time: ", len(timesteps) * dt)
    #     for j in range(1, len(timesteps)):
    #         if j % 100 == 0:
    #             plt.plot(
    #                 torch.linspace(0, length, n),
    #                 timesteps[j],
    #                 label="t = %.5f s" % (j * dt),
    #             )

    #     plt.title("1D heat equation in a rod of length 1")
    #     plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    #     plt.grid(True)
    #     plt.xlabel("Length", fontsize=14)
    #     plt.ylabel("Temperature", fontsize=14)
    #     plt.tight_layout()
    #     plt.show()

    #     print("dim: ", len(timesteps[-1]))

    return timesteps[-1] - torch.ones(30) * 10


def heat_equation_pytorch_jacobian(input):
    return jacfwd(heat_equation_pytorch)(input)


def rosenbrock(input: np.ndarray, a=1, b=100):
    return ((a - input[0]) ** 2) + b * ((input[1] - input[0] ** 2) ** 2)


def rosenbrock_gradient(input: np.ndarray, a=1, b=100):
    dx = (-2 * (a - input[0])) - (4 * b * input[0] * (input[1] - input[0] ** 2))
    dy = 2 * b * (input[1] - input[0] ** 2)

    return np.array([dx, dy])


def rosenbrock_hessian(input: np.ndarray, b=100):
    dxx = (12 * b * input[0] ** 2) - (4 * b * input[1]) + 2
    dxy = -4 * b * input[0]
    dyx = -4 * b * input[0]
    dyy = 2 * b

    return np.array([[dxx, dxy], [dyx, dyy]])


def rosenbrock_gradient_surrogate(input: torch.Tensor):
    return model(input)


def test_function(input):
    A = torch.tensor([[1.0, 0.0, 0.0], [4.0, 5.0, 0.0], [7.0, 8.0, 9.0]])

    B = torch.tensor([[0.5, 1.0, 1.5], [0.0, 2.5, 3.0], [0.0, 0.0, 4.5]])

    C = torch.tensor([[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [7.0, 2.0, 4.0]])

    D = torch.tensor([[2.0, 8.0, 3.0], [0.0, 3.0, 5.0], [0.0, 0.0, 4.0]])

    i1 = torch.matmul(D, input)
    i2 = torch.matmul(C, i1)
    i3 = torch.matmul(B, i2)
    i4 = torch.matmul(A, i3)

    return i4 - torch.tensor([14573, 199447, 639790])


def test_function_jacobian(input):
    return jacfwd(test_function)(input)


class FindRoot:
    def __init__(self, f, Jf, threshold=1e-6, max_iter=10000):
        self.f = f
        self.Jf = Jf
        self.threshold = threshold
        self.max_iter = max_iter
        self.cgb = ComputationalGrapBuilder()

    def newton_raphson(self, initial_guess):
        p = torch.tensor(initial_guess)
        iterno = 0
        parray = [p]
        fprev = self.f(p)
        farray = [fprev]

        while iterno < self.max_iter:
            f = self.f(torch.tensor(p))
            j = self.Jf(torch.tensor(p))

            delta = torch.linalg.solve(j, -f)
            p = torch.tensor(p + delta)

            fcur = f

            if torch.any(torch.isnan(fcur)):
                break

            parray.append(p)
            farray.append(fcur)

            print("iteration: ", iterno, ", d: ", torch.linalg.norm(delta))
            if torch.linalg.norm(delta) < self.threshold:
                break
            fprev = fcur
            iterno += 1

        return parray, farray

    def matrix_free_newton(self, initial_guess):
        p = initial_guess
        iterno = 0
        parray = []
        fprev = self.f(initial_guess)
        farray = [fprev]

        while iterno < self.max_iter:
            f = self.f(torch.tensor(p, requires_grad=True))

            matrix_chain = self.cgb.construct_graph(f)
            z = torch.tensor(-f)
            i = 0

            for matrix in matrix_chain:
                z = torch.linalg.solve(matrix, z)

            i += 1

            p = torch.tensor(p + z)

            fcur = self.f(p)

            if torch.any(torch.isnan(fcur)):
                break

            parray.append(p)
            farray.append(fcur)

            print("iteration: ", iterno)
            print("z: ", torch.norm(z))

            if torch.norm(z) < self.threshold:
                break

            fprev = fcur
            iterno += 1

        return parray, farray


findRoot = FindRoot(heat_equation_pytorch, heat_equation_pytorch_jacobian)

# input = torch.rand(100) * 100
# input[0] = 1.0
# input[-1] = 1.0

# output = heat_equation_pytorch(input)

# print(torch.linalg.norm(output - torch.ones(5)))

# output_pytorch = heat_equation_pytorch(input_pytorch)

# print("input: ", input_pytorch)
# print("output: ", output_pytorch)


# input_pytorch = torch.rand(101, requires_grad=True)

# parray, farray = findRoot.newton_raphson(initial_guess=torch.rand(3) * 10)

# print(parray[-1])
# print(farray[-1])


# print("Number of Iterations: ", len(farray))

input = torch.rand(30).double() * 10

input[0] = 1.0
input[-1] = 1.0


parray, farray = findRoot.newton_raphson(initial_guess=input)

print(parray[-1])
print(farray[-1])

print("Number of Iterations: ", len(farray))
