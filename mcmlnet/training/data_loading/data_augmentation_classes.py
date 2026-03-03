"""
Collection of data augmentation methods, wraps around the augmentation functions
defined in data_augmentation.py to allow for easy instantiation and application
of augmentations to input and output data in a training loop.
"""

from abc import ABC, abstractmethod

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import RandomOrder

from mcmlnet.training.data_loading.data_augmentation import (
    add_noise,
    add_shot_noise,
    bezier_contrast,
    brightness_dark_variation,
)
from mcmlnet.utils.logging import setup_logging

logger = setup_logging(level="info", logger_name=__name__)


class DataAugmentation(ABC):
    """Abstract base class for all data augmentation classes."""

    def __init__(self, p: float = 1.0) -> None:
        """Initialize base augmentation class.

        Args:
            p: Probability that transform is applied.
        """
        if not isinstance(p, int | float) or p < 0.0 or p > 1.0:
            raise ValueError(
                f"Probability 'p' must be a number between 0 and 1, got {p}"
            )
        self.p = p

    @abstractmethod
    def __call__(self, x: torch.Tensor | None) -> torch.Tensor | None:
        """Apply augmentation to input data."""


class NoiseAddition(DataAugmentation):
    """Add noise to data with specified signal-to-noise ratio."""

    def __init__(self, snr: float, p: float = 1.0, clipping: bool = True) -> None:
        """Initialize noise addition augmentation.

        Args:
            snr: Fraction of noise to add (NOT log SNR but plain quotient).
            p: Probability that transform is applied.
            clipping: Whether to clip the tensor back to range [0, 1].
        """
        super().__init__(p)
        if not isinstance(snr, int | float) or isinstance(snr, bool):
            raise TypeError(f"SNR must be a numerical type, got {type(snr)}")
        if snr <= 0:
            raise ValueError(f"SNR must be positive, got {snr}")

        self.snr = snr
        self.clipping = clipping

    def __call__(self, x: torch.Tensor | None) -> torch.Tensor | None:
        """Apply noise addition augmentation."""
        if x is None:
            return None
        return add_noise(x, self.snr, self.p, clipping=self.clipping)


class NoiseAdditionWithExponentialDecay(NoiseAddition):
    """Add noise with exponential decay over training epochs."""

    def __init__(
        self,
        snr: float,
        max_epochs: int,
        batches_per_epoch: int,
        p: float = 1.0,
        clipping: bool = True,
        decay_scale: float = 5.0,
    ) -> None:
        """
        Initialize noise addition with exponential decay.

        Implements add_noise with exponential decay schedule.
        snr is NOT log SNR but plain noise ratio.

        Args:
            snr: Fraction of noise to add.
            max_epochs: Maximum number of training epochs.
            batches_per_epoch: Number of batches per epoch.
            p: Probability that transform is applied.
            clipping: Whether to clip the tensor back to range [0, 1].
            decay_scale: Scale factor for exponential decay.
        """
        super().__init__(snr, p, clipping)
        if batches_per_epoch <= 0:
            raise ValueError(
                f"batches_per_epoch must be positive, got {batches_per_epoch}"
            )
        if decay_scale <= 0:
            raise ValueError(f"decay_scale must be positive, got {decay_scale}")

        self.current_epoch: int = 0
        self.current_batch: int = 0
        self.max_epochs = max_epochs
        self.batches_per_epoch = batches_per_epoch
        self.decay_scale = torch.tensor([decay_scale])

    def decay_schedule(self) -> float:
        """Compute the current exponential decay schedule factor.

        Returns:
            Current decay factor based on epoch progress.
        """
        return float(
            torch.exp(-self.decay_scale * self.current_epoch / self.max_epochs)
        )

    def __call__(self, x: torch.Tensor | None) -> torch.Tensor | None:
        """Apply noise addition with exponential decay."""
        if x is None:
            return None

        # Keep track of the current epoch and batch
        self.current_batch += 1
        if self.current_batch >= self.batches_per_epoch:
            self.current_batch = 0
            self.current_epoch += 1

        # Compute the current noise level
        current_snr = self.snr / self.decay_schedule()

        # Log progress periodically
        if (
            self.current_batch == 0
            and self.current_epoch % max(self.max_epochs // 20, 1) == 0
        ):
            logger.info(
                f"Current epoch: {self.current_epoch}, batch: {self.current_batch}"
            )
            logger.info(f"Base SNR: {self.snr}, Current SNR: {current_snr}")

        return add_noise(x, current_snr, self.p, clipping=self.clipping)


class ReflectanceMCNoise(NoiseAddition):
    """Simulate Monte Carlo noise for reflectance data."""

    def __init__(self, n_photons: int, p: float = 1.0, clipping: bool = True) -> None:
        """Initialize Monte Carlo noise simulation.

        Implements noise similar to MC uncertainty, defining the snr to emulate
        a standard deviation of sqrt(p*(1-p) / n) (Binomial reflectance std).

        Args:
            n_photons: Amount of photons used for the simulation.
            p: Probability that transform is applied.
            clipping: Whether to clip the tensor back to range [0, 1].
        """
        if n_photons <= 0:
            raise ValueError(f"n_photons must be positive, got {n_photons}")

        self.n_photons = n_photons
        # Use dummy SNR as it will be computed dynamically
        super().__init__(snr=1.0, p=p, clipping=clipping)

    def __call__(self, x: torch.Tensor | None) -> torch.Tensor | None:
        """Apply Monte Carlo noise simulation."""
        if x is None:
            return None

        if not (x > 0.0).all() or not (x < 1.0).all():
            raise ValueError("Input data needs to be in range (0, 1)")
        std = torch.sqrt(x * (1 - x) / self.n_photons)

        return add_noise(x, 1 / std, self.p, clipping=self.clipping)


class ShotNoiseAddition(DataAugmentation):
    """Add shot noise to data based on white and dark reference measurements."""

    def __init__(
        self,
        white: torch.Tensor | None = None,
        dark: torch.Tensor | None = None,
        snr: float = 20.0,
        p: float = 1.0,
        clipping: bool = True,
    ) -> None:
        """Initialize shot noise addition.

        Args:
            white: White reference measurement, has to have the same number
              of wavelengths as the image or spectrum it will be applied to.
            dark: Dark reference measurement, has to have the same number
              of wavelengths as the image or spectrum it will be applied to.
            snr: Ratio of white / dark (not exactly the full SNR!).
            p: Probability that noise is applied.
            clipping: Whether to clip data to range [0,1].
        """
        super().__init__(p)
        if not isinstance(snr, int | float) or isinstance(snr, bool):
            raise TypeError(f"SNR must be a numerical type, got {type(snr)}")
        if snr <= 0:
            raise ValueError(f"SNR must be positive, got {snr}")
        if white is not None and not isinstance(white, torch.Tensor):
            raise TypeError(f"white must be torch.Tensor, got {type(white)}")
        if dark is not None and not isinstance(dark, torch.Tensor):
            raise TypeError(
                f"dark must be torch.Tensor or np.ndarray, got {type(dark)}"
            )

        self.white = white
        self.dark = dark
        self.snr = snr
        self.clipping = clipping

    def __call__(self, x: torch.Tensor | None) -> torch.Tensor | None:
        """Apply shot noise addition."""
        if x is None:
            return None
        return add_shot_noise(
            x, self.white, self.dark, self.snr, self.p, clipping=self.clipping
        )


class BrightDarkVariation(DataAugmentation):
    """Apply bright and dark level variations to data."""

    def __init__(self, p: float = 1.0, eps: float = 0.2) -> None:
        """Initialize bright and dark level variations.

        Args:
            p: Probability that transform is applied.
            eps: Scale of the uniform distribution to sample spectral "squeeze" from.
        """
        super().__init__(p)
        if not isinstance(eps, int | float) or isinstance(eps, bool):
            raise TypeError(f"eps must be a numerical type, got {type(eps)}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.eps = eps

    def __call__(self, x: torch.Tensor | None) -> torch.Tensor | None:
        """Apply bright and dark level variations."""
        if x is None:
            return None
        return brightness_dark_variation(x, eps=self.eps, p=self.p)


class BezierContrast(DataAugmentation):
    """Apply Bezier curve contrast transformation to data."""

    def __init__(self, p: float = 1.0, p_flip: float = 0.2) -> None:
        """Initialize Bezier curve contrast transformation.

        Args:
            p: Probability that transform is applied.
            p_flip: Probability for "flipped" contrasts
                (Bezier curve going from (1,1) -> (0,0)).
        """
        super().__init__(p)
        if not isinstance(p_flip, int | float) or isinstance(p_flip, bool):
            raise TypeError(f"p_flip must be a numerical type, got {type(p_flip)}")
        if p_flip < 0.0 or p_flip > 1.0:
            raise ValueError(f"p_flip must be between 0 and 1, got {p_flip}")

        self.p_flip = p_flip

    def __call__(self, x: torch.Tensor | None) -> torch.Tensor | None:
        """Apply Bezier curve contrast transformation."""
        if x is None:
            return None
        return bezier_contrast(x, p=self.p, p_flip=self.p_flip)


def instantiate_augmentations(kwargs: DictConfig) -> tuple[RandomOrder, RandomOrder]:
    """
    Wrap around data_augmentation to augment optical parameters and
    reflectance for better robustness.

    Args:
        kwargs: Configuration containing augmentation parameters:
            - 'snr' (float): fraction of noise to add
            - 'p_noise' (float): probability that noise is applied
            - 'p_bright_dark' (float): probability that simple brightness and
                contrast scaling is applied
            - 'eps' (float): reflectance re-scaling parameter
            - 'p_bezier' (float): probability that bezier brightness and
                contrast scaling is applied

    Returns:
        Tuple of randomly ordered augmentations for input and output data.
    """
    # OmegaConf only supports primitive types, thus convert to regular dict
    # to create list of transforms
    input_list = list(
        OmegaConf.to_container(
            instantiate(kwargs.input_augmentations), resolve=True
        ).values()
    )
    output_list = list(
        OmegaConf.to_container(
            instantiate(kwargs.output_augmentations), resolve=True
        ).values()
    )

    # Use RandomOrder instead of Compose
    trans_input = RandomOrder(input_list)
    trans_output = RandomOrder(output_list)

    return trans_input, trans_output


def apply_augmentations(
    input: torch.Tensor,
    output: torch.Tensor,
    input_augmentations: RandomOrder,
    output_augmentations: RandomOrder,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply augmentations to optical parameters and reflectance for better robustness.

    Args:
        input: Optical parameters which were used in the MC simulation.
        output: MC simulated reflectance spectra, expected to be in [0,1].
        input_augmentations: Augmentations to apply to input data.
        output_augmentations: Augmentations to apply to output data.

    Returns:
        Tuple of augmented input and output data.
    """
    return input_augmentations(input), output_augmentations(output)
