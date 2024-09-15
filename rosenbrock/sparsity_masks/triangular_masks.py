import torch


def UpperTriangularMask(dim: int) -> torch.Tensor:
    upper_triangular_matrix = torch.zeros(dim, dim)
    for i in range(dim):
        for j in range(dim):
            if i <= j:
                upper_triangular_matrix[i, j] = 1.0
    return upper_triangular_matrix


def LowerTriangularMask(dim: int) -> torch.Tensor:
    lower_triangular_matrix = torch.zeros(dim, dim)
    for i in range(dim):
        for j in range(dim):
            if i >= j:
                lower_triangular_matrix[i, j] = 1.0
    return lower_triangular_matrix
