import numpy as np
from matplotlib import pyplot as plt

from rosenbrock.generate_data.function import rosenbrock_nd_gradient

x = np.linspace(-2, 2, 20)
y = np.linspace(-1, 3, 20)
X, Y = np.meshgrid(x, y)


U = np.zeros_like(X)
V = np.zeros_like(Y)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        grad = rosenbrock_nd_gradient(np.array([X[i, j], Y[i, j]]))
        U[i, j] = grad[0]
        V[i, j] = grad[1]

plt.figure(figsize=(8, 6))
plt.quiver(X, Y, U, V, color="r")
plt.title("Gradient Field of Rosenbrock Function")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
