import numpy as np


def rosenbrock(input: np.ndarray, a=1, b=100):
    return ((a - input[0]) ** 2) + b * ((input[1] - input[0] ** 2) ** 2)


def rosenbrock_gradient(input: np.ndarray, a=1, b=100):
    dx = (-2 * (a - input[0])) - (4 * b * input[0] * (input[1] - input[0] ** 2))
    dy = 2 * b * (input[1] - input[0] ** 2)

    return np.array([dx, dy])
