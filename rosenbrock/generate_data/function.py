import numpy as np


def rosenbrock(input: np.ndarray, a=1, b=100):
    return ((a - input[0]) ** 2) + b * ((input[1] - input[0] ** 2) ** 2)


def rosenbrock_gradient(input: np.ndarray, a=1, b=100):
    dx = (-2 * (a - input[0])) - (4 * b * input[0] * (input[1] - input[0] ** 2))
    dy = 2 * b * (input[1] - input[0] ** 2)

    return np.array([dx, dy])


def rosenbrock_nd(input: np.ndarray, a=1, b=100):
    n = len(input)
    return sum(
        (a - input[i]) ** 2 + b * (input[i + 1] - input[i] ** 2) ** 2
        for i in range(n - 1)
    )


def rosenbrock_nd_gradient(input: np.ndarray, a=1, b=100):
    n = len(input)
    gradient = np.zeros_like(input)

    for i in range(n - 1):
        gradient[i] += -2 * (a - input[i]) - 4 * b * input[i] * (
            input[i + 1] - input[i] ** 2
        )
        if i + 1 < n:
            gradient[i + 1] += 2 * b * (input[i + 1] - input[i] ** 2)

    return gradient
