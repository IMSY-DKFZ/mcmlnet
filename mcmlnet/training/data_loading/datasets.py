"""
Common place for all the different datasets.
"""

import torch
from torch.utils.data import Dataset

from mcmlnet.training.data_loading.preprocessing import (
    PreProcessor,
    process_2d_3d_data,
    set_deepest_layer_to_zero,
)
from mcmlnet.utils.logging import setup_logging

logger = setup_logging(level="info", logger_name=__name__)


def mix_spectra_randomly(spectra: torch.Tensor, mix_data: bool) -> None:
    """
    Break-up the order of the spectra in the tensor
    to avoid batch-wise similar physiological values.

    Args:
        spectra: 3d (batch, lambda, param.s) tensor containing the MC spectral data.
        mix_data: Whether to mix the data randomly.

    Raises:
        TypeError: If the input tensor is not a torch.Tensor.
        ValueError: If the input tensor is not 3-dimensional or has invalid shape.
    """
    if not isinstance(spectra, torch.Tensor):
        raise TypeError(f"spectra must be torch.Tensor, got {type(spectra)}")

    if spectra.ndim != 3:
        raise ValueError(f"Input tensor must be 3-dimensional, got {spectra.ndim}D")

    if spectra.shape[0] == 0 or spectra.shape[1] == 0 or spectra.shape[2] == 0:
        raise ValueError("Input tensor must have non-zero dimensions")

    if mix_data:
        # Fix seeds for reproducibility (for each wavelength one seed)
        torch.manual_seed(42)
        seeds = torch.randperm(1000)[: spectra.size(1)]

        for i in range(spectra.size(1)):
            torch.manual_seed(seeds[i])
            idcs = torch.randperm(spectra.size(0)).to(spectra.device)
            spectra[:, i, :] = spectra[idcs, i, :]


class DatasetMCPredDirect(Dataset):
    """Turn the simulation Tensor into a dataset for PyTorch."""

    def __init__(
        self,
        data: torch.Tensor,
        n_wavelengths: int,
        n_params: int,
        thick_deepest_layer: bool = True,
        mix_data: bool = True,
        preprocessor: PreProcessor | None = None,
    ) -> None:
        """Initialize MC dataset in form of a torch.Tensor.

        Args:
            data: 3d tensor containing the MC spectral data and physical parameters.
            n_wavelengths: Amount of wavelengths in 'data'.
            n_params: Amount of (physical) parameters per wavelength.
            thick_deepest_layer: Whether deepest layer is very thick,
                sets values to zero to avoid artifacts.
            mix_data: Whether to mix spectra randomly.
            preprocessor: Preprocessor to apply to the data.

        Raises:
            ValueError: If parameters are invalid or
                data shape does not match expectations.
            TypeError: If data is not a torch.Tensor.
        """
        super().__init__()

        if not isinstance(data, torch.Tensor):
            raise TypeError(f"data must be torch.Tensor, got {type(data)}")

        if not isinstance(n_wavelengths, int) or n_wavelengths <= 0:
            raise ValueError(
                f"n_wavelengths must be positive integer, got {n_wavelengths}"
            )

        if not isinstance(n_params, int) or n_params <= 0:
            raise ValueError(f"n_params must be positive integer, got {n_params}")

        if not isinstance(thick_deepest_layer, bool):
            raise TypeError(
                f"thick_deepest_layer must be bool, got {type(thick_deepest_layer)}"
            )

        if not isinstance(mix_data, bool):
            raise TypeError(f"mix_data must be bool, got {type(mix_data)}")

        self.data = data
        self.n_params = n_params
        self.n_wavelengths = n_wavelengths
        self.thick_layer = thick_deepest_layer
        self.preprocessor = preprocessor

        if self.data.ndim == 3:
            expected_shape = (n_wavelengths, n_params + 1)
            if self.data[0].shape != expected_shape:
                raise ValueError(
                    f"Individual data required to be of shape {expected_shape}, "
                    f"got {self.data[0].shape}"
                )
            mix_spectra_randomly(self.data, mix_data)
        else:
            logger.warning(
                "Continuing with dummy 2D tensor for dataset init. "
                "Make sure this is intended!"
            )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return int(self.data.shape[0])

    def __getitem__(
        self, index: int | list[int] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get item by index.

        Args:
            index: Index of the item to retrieve.

        Returns:
            Tuple of (parameters, reflectance) tensors.

        Raises:
            TypeError: If index is not an integer.
            IndexError: If index is out of bounds.
            ValueError: If preprocessor is not provided.
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor is required but not provided")

        data = process_2d_3d_data(self.data, index, self.preprocessor)
        data = set_deepest_layer_to_zero(data, self.thick_layer)

        params = data[..., : self.n_params].reshape(-1, self.n_params)
        reflectance = (
            data[..., self.n_params :].reshape(-1, self.n_wavelengths).squeeze()
        )

        return params, reflectance


class DatasetMCPred(DatasetMCPredDirect):
    """Load the data lazily and preprocess it on the fly in collate_fn."""

    def __getitem__(self, index: int | list[int] | torch.Tensor) -> torch.Tensor:
        return self.data[index]
