import torch


def min_max_normalise(
    input: torch.Tensor, min: torch.Tensor | float, max: torch.Tensor | float
):
    return (input - min) / (max - min)


def min_max_denormalise(
    input: torch.Tensor, min: torch.Tensor | float, max: torch.Tensor | float
):
    return (input * (max - min)) + min
