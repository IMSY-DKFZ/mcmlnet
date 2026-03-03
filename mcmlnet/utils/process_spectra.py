"""Utility functions for processing reflectance spectra."""

import numpy as np
import torch

from mcmlnet.utils.tensor import TensorType, tensor_conversion_decorator


def r_specular(
    n1: float | torch.Tensor | np.ndarray, n2: float | torch.Tensor | np.ndarray
) -> float | torch.Tensor | np.ndarray:
    """
    Computes the specular reflectance to be added
    to a multi-layer tissue model reflectance.

    Args:
        n1: refractive index of the top medium (usually air)
        n2: refractive index of the bottom medium (usually first tissue layer)

    Returns:
        Specular reflectance.
    """
    return (n1 - n2) ** 2 / (n1 + n2) ** 2


@tensor_conversion_decorator
def coeff_of_variation(data: TensorType, n_photons: float) -> TensorType:
    """
    Calculate the coefficient of variation for reflectance data.

    Args:
        data: Reflectance data.
        n_photons: Number of photons.

    Returns:
        Coefficient of variation for each column.

    Raises:
        ValueError: If data is not in the range [0, 1] or if n_photons is not positive.
    """
    if torch.any(data < 0) or torch.any(data > 1):
        raise ValueError("Data must be in the range [0, 1]")
    if n_photons <= 0:
        raise ValueError("Number of photons must be greater than zero")

    return torch.sqrt((1 - data) / data / n_photons)
