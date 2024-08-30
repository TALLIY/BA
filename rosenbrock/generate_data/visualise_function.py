import numpy as np
from matplotlib import pyplot as plt

from rosenbrock.generate_data.function import rosenbrock

x = np.linspace(-3, 3, 2000)
y = np.linspace(-4, 9, 2000)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Visualization of the Rosenbrock Function")
plt.show()

plt.figure(figsize=(10, 7))
plt.contourf(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap="viridis")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Contour Plot of the Rosenbrock Function")
plt.show()
