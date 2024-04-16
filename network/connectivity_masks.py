import numpy as np


def upper_traingular_mask(input_dim, output_dim):
    upper_traingular_matrix = np.full(shape=(input_dim, output_dim), fill_value=0)
    for i in range(input_dim):
        for j in range(output_dim):
            if i >= j:
                upper_traingular_matrix[i][j] = 1.0

    return upper_traingular_matrix


def lower_traingular_mask(input_dim, output_dim):
    lower_traingular_matrix = np.full(shape=(input_dim, output_dim), fill_value=0)
    for i in range(input_dim):
        for j in range(output_dim):
            if i <= j:
                lower_traingular_matrix[i][j] = 1.0
    return lower_traingular_matrix
