from sparse_layer import sparse_layer
from connectivity_masks import upper_traingular_mask, lower_traingular_mask


def upper_traingular_layer(input_dim, output_dim):
    layer = sparse_layer(
        input_dim, output_dim, upper_traingular_mask(input_dim, output_dim)
    )
    return layer


def lower_traingular_layer(input_dim, output_dim):
    layer = sparse_layer(
        input_dim, output_dim, lower_traingular_mask(input_dim, output_dim)
    )
    return layer
