"""Collection of data augmentation methods."""

import numpy as np
import torch

from mcmlnet.utils.logging import setup_logging
from mcmlnet.utils.tensor import TensorType

logger = setup_logging(level="info", logger_name=__name__)


def _sample_selection_mask(x: TensorType, p: float) -> TensorType:
    """
    Create a bool mask along the batch axis
    for individual sample augmentation selection.

    Args:
        x: Data to augment
        p: Probability that augmentation is applied

    Returns:
        Boolean mask for sample selection

    Raises:
        TypeError: If x is not a torch.Tensor or np.ndarray
        ValueError: If p is not in [0, 1]
    """
    if not isinstance(p, int | float) or p < 0.0 or p > 1.0:
        raise ValueError(f"Probability p must be between 0 and 1, got {p}")

    if not isinstance(x, torch.Tensor | np.ndarray):
        raise TypeError(f"x must be torch.Tensor or np.ndarray, got {type(x)}")

    if x.shape[0] == 0:
        raise ValueError("Input tensor/array must have at least one sample")

    p_mask = torch.rand(x.shape[0]) < p

    if isinstance(x, torch.Tensor):
        return p_mask.to(device=x.device)
    else:
        return p_mask.numpy()


def add_noise(
    x: TensorType, snr: float | torch.Tensor, p: float = 1.0, clipping: bool = True
) -> TensorType:
    """Add noise to the data.

    Args:
        x: Data to augment.
        snr: Fraction of noise to add to intensity of the signal.
        p: Probability that noise is applied.
        clipping: Whether to clip data to range [0,1].

    Returns:
        Augmented data.

    Raises:
        TypeError: If x is not a torch.Tensor or np.ndarray
        ValueError: If snr is not positive or p is not in [0, 1]
    """
    if isinstance(snr, torch.Tensor):
        if (snr <= 0).any():
            raise ValueError(f"All SNR values must be positive, got {snr}")
        if not snr.shape == x.shape:
            raise ValueError(f"SNR shape must match input shape, got {snr.shape}")
    else:
        if not isinstance(snr, int | float) or snr <= 0:
            raise ValueError(f"SNR must be positive, got {type(snr)}: {snr}")

    if not isinstance(x, torch.Tensor | np.ndarray):
        raise TypeError(f"x must be torch.Tensor or np.ndarray, got {type(x)}")

    if not isinstance(p, int | float) or p < 0.0 or p > 1.0:
        raise ValueError(f"Probability p must be between 0 and 1, got {p}")

    p_mask = _sample_selection_mask(x, p)

    if isinstance(x, torch.Tensor):
        # According to https://discuss.pytorch.org/t/random-number-generation-speed/12209,
        # normal_ is the fastest way, compared to FloatTensor().normal or randn_like
        x_aug = x * (1 + x.new(size=(x.shape)).normal_() / snr)
        x_aug = torch.clip(x_aug, min=0, max=1) if clipping else x_aug
    else:
        x_aug = x * (1 + np.random.randn(*x.shape) / snr)
        x_aug = np.clip(x_aug, a_min=0, a_max=1) if clipping else x_aug

    # Overwrite specific samples with their augmented counterparts
    x[p_mask] = x_aug[p_mask]

    return x


def add_shot_noise(
    x: TensorType,
    white: TensorType | None = None,
    dark: TensorType | None = None,
    snr: float = 20.0,
    p: float = 1.0,
    clipping: bool = True,
) -> TensorType:
    """Add shot noise to the data.

    Args:
        x: Data to augment.
        white: White reference measurement, has to have the same number
              of wavelengths as the image or spectrum it will be applied to.
        dark: Dark reference measurement, has to have the same number
              of wavelengths as the image or spectrum it will be applied to.
        snr: Ratio of white / dark (not exactly the full SNR!).
        p: Probability that noise is applied.
        clipping: Whether to clip data to range [0,1].

    Returns:
        Augmented data.

    Raises:
        TypeError: If x is not a torch.Tensor or np.ndarray
        ValueError: If snr is not positive or p is not in [0, 1]
    """
    if not isinstance(snr, int | float) or snr <= 0:
        raise ValueError(f"SNR must be positive, got {snr}")

    if not isinstance(x, torch.Tensor | np.ndarray):
        raise TypeError(f"x must be torch.Tensor or np.ndarray, got {type(x)}")

    if not isinstance(white, torch.Tensor | np.ndarray):
        raise TypeError(f"white must be torch.Tensor or np.ndarray, got {type(white)}")

    if not isinstance(dark, torch.Tensor | np.ndarray):
        raise TypeError(f"dark must be torch.Tensor or np.ndarray, got {type(dark)}")

    if not isinstance(p, int | float) or p < 0.0 or p > 1.0:
        raise ValueError(f"Probability p must be between 0 and 1, got {p}")

    p_mask = _sample_selection_mask(x, p)
    ones_like_x = torch.ones_like(x) if isinstance(x, torch.Tensor) else np.ones_like(x)

    # Apply SNR approach (approximation)
    if (white is None) or (dark is None):
        if snr <= 0:
            return x
        # Assume white to be uniform when no white and/ or dark are given
        assumed_white = 10000 * ones_like_x if white is None else white
        assumed_dark = assumed_white / snr if dark is None else dark
        # Calculate noise variance
        variance = (x + x * x) / (assumed_white - assumed_dark)

    # Apply white / dark reference approach
    else:
        # Calculate noise variance
        variance = (x + x * x) / (white - dark)

    if isinstance(x, torch.Tensor):
        # According to https://discuss.pytorch.org/t/random-number-generation-speed/12209,
        # normal_ is the fastest way, compared to FloatTensor().normal or randn_like
        x_aug = x + torch.rand_like(x) * torch.sqrt(variance)
        x_aug = torch.clip(x_aug, min=0, max=1) if clipping else x_aug
    else:
        x_aug = x + np.random.randn(*x.shape) * np.sqrt(variance)
        x_aug = np.clip(x_aug, a_min=0, a_max=1) if clipping else x_aug

    # Overwrite specific samples with their augmented counterparts
    x[p_mask] = x_aug[p_mask]

    return x


def brightness_dark_variation(x: TensorType, eps: float, p: float = 1.0) -> TensorType:
    """Apply brightness and dark variation to the data.

    Input x is rescaled like x' = (x + s1) / ((1 + s1)*(1 + s2)), using s sampled from
    the uniform distribution s ~ eps * U[0,1).

    Args:
        x: Data to augment.
        eps: Scale of the uniform distribution to sample spectral "squeeze" from.
        p: Probability that bright and dark values are changed.

    Returns:
        Augmented data.

    Raises:
        TypeError: If x is not a torch.Tensor or np.ndarray
        ValueError: If eps is not positive or p is not in [0, 1]
    """
    if not isinstance(eps, int | float) or eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    if not isinstance(x, torch.Tensor | np.ndarray):
        raise TypeError(f"x must be torch.Tensor or np.ndarray, got {type(x)}")

    if not isinstance(p, int | float) or p < 0.0 or p > 1.0:
        raise ValueError(f"Probability p must be between 0 and 1, got {p}")

    p_mask = _sample_selection_mask(x, p)

    if isinstance(x, torch.Tensor):
        s1 = x.new(size=(x.shape[0], 1)).uniform_() * eps
        x_aug = (x + s1) / (
            (1 + s1) * (1 + eps * x.new(size=(x.shape[0], 1)).uniform_() * eps)
        )

    else:
        s1 = np.random.rand(x.shape[0], 1) * eps
        s2 = np.random.rand(x.shape[0], 1) * eps
        x_aug = (x + s1) / ((1 + s1) * (1 + s2))

    # Overwrite specific samples with their augmented counterparts
    x[p_mask] = x_aug[p_mask]

    return x


def bezier_contrast(x: TensorType, p: float = 1.0, p_flip: float = 0.5) -> TensorType:
    """Apply Bezier curve contrast transformation.

    Transform the intensity as described e.g. in https://arxiv.org/pdf/2004.07882.pdf.

    Args:
        x: Data to augment.
        p: Probability that Bezier curves are applied to change contrast.
        p_flip: Probability for "flipped" contrasts
            (Bezier curve going from (1,1) -> (0,0)).

    Returns:
        Augmented data.

    Raises:
        TypeError: If x is not a torch.Tensor or np.ndarray
        ValueError: If p or p_flip are not in [0, 1]
    """
    if not isinstance(p, int | float) or p < 0.0 or p > 1.0:
        raise ValueError(f"Probability p must be between 0 and 1, got {p}")

    if not isinstance(x, torch.Tensor | np.ndarray):
        raise TypeError(f"x must be torch.Tensor or np.ndarray, got {type(x)}")

    if not isinstance(p_flip, int | float) or p_flip < 0.0 or p_flip > 1.0:
        raise ValueError(f"Probability p_flip must be between 0 and 1, got {p_flip}")

    p_mask = _sample_selection_mask(x, p)

    # Set Bezier curve endpoints
    if torch.rand(1)[0] < p_flip:
        p0, p3 = 1, 0
    else:
        p0, p3 = 0, 1

    # Select control points from within [0,1]
    p1, p2 = torch.rand(1)[0], torch.rand(1)[0]

    # Remember original type and convert to torch for computation
    is_numpy = isinstance(x, np.ndarray)
    if is_numpy:
        x = torch.from_numpy(x)

    # Bezier curve "mapping"
    x_aug = (
        (1 - x) ** 3 * p0
        + 3 * (1 - x) ** 2 * x * p1
        + 3 * (1 - x) * x**2 * p2
        + x**3 * p3
    )

    # Convert back to numpy if input was numpy
    if is_numpy:
        x_aug = x_aug.numpy()
        x = x.numpy()

    # Overwrite specific samples with their augmented counterparts
    x[p_mask] = x_aug[p_mask]

    return x
