import torch

from network.computational_graph_builder import ComputationalGrapBuilder
from network.ftcs_pytorch import FTCS

# torch.set_printoptions(threshold=float("inf"))


sim = FTCS(length=1, n=101, alpha=0.01, min_T=0, max_T=100)
timesteps, X = sim.finite_difference(number_of_iterations=1000, graph=True)

print(timesteps[-1])

cgb = ComputationalGrapBuilder()
chain = cgb.construct_graph(timesteps[-1])

print("chain length: ", len(chain))

print(
    torch.linalg.lu_factor(
        torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).float(), False
    )
)


# def matrix_free_newton_method(matrix_chain):
#     with torch.no_grad():
#         for matrix in chain:
