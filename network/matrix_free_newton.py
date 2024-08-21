from network.ftcs import FTCS

sim = FTCS(length=1, n=101, alpha=0.01, min_T=0, max_T=100)
_, X = sim.finite_difference(number_of_iterations=1000, graph=True)
