from computational_graph_builder import ComputationalGrapBuilder
from ftcs import FTCS

sim = FTCS(length=1, n=101, alpha=0.01, min_T=0, max_T=100)
timesteps, A = sim.finite_difference(number_of_iterations=1000, graph=True)

print(A)

cgb = ComputationalGrapBuilder()
chain = cgb.construct_graph(timesteps[-1])

print("matrix")
for matrix in chain:
    print(matrix)
