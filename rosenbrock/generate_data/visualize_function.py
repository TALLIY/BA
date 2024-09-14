import numpy as np
from matplotlib import pyplot as plt

from rosenbrock.generate_data.function import rosenbrock, rosenbrock_gradient

x = np.linspace(-2, 2, 30)
y = np.linspace(-1, 3, 30)
X, Y = np.meshgrid(x, y)

Z = np.array([rosenbrock(np.array([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))])
Z = Z.reshape(X.shape)

plt.figure(figsize=(10, 7))
contour = plt.contourf(X, Y, Z, levels=np.logspace(-1, 3, 100), cmap="viridis")
plt.colorbar(contour)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Contour Plot of the Rosenbrock Function")
plt.show()

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")
surface = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Surface Plot of the Rosenbrock Function")
plt.colorbar(surface)
plt.show()

x = np.linspace(-2, 2, 30)
y = np.linspace(-1, 3, 30)
X, Y = np.meshgrid(x, y)

Z = rosenbrock(X, Y)

U = np.zeros_like(X)
V = np.zeros_like(Y)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        grad = rosenbrock_gradient(np.array([X[i, j], Y[i, j]]))
        U[i, j] = grad[0]
        V[i, j] = grad[1]

plt.figure(figsize=(10, 7))
contour = plt.contourf(X, Y, Z, levels=50, cmap="viridis")
plt.colorbar(contour)

plt.quiver(X, Y, U, V, color="white")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Rosenbrock Function with Gradient Vector Field")

plt.show()
