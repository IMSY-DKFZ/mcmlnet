"""Simplified preprocessor and Dataset for SOTA datasets."""

import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from mcmlnet.experiments.data_loaders.simulation import ManojlovicSimulationLoader
from mcmlnet.utils.logging import setup_logging

logger = setup_logging(level="info", logger_name=__name__)

BASE_DIR = os.path.join(os.environ["data_dir"], "raw/related_work_reimplemented/")


class SOTAPreprocessor:
    """Simplified preprocessor for SOTA datasets."""

    def __init__(
        self,
        dataset_name: str,
        n_wavelengths: int,
        log_intensity: bool = False,
        n_pca_comp: int = 0,
        kfolds: int | None = None,
        fold: int | None = None,
        norm_1: torch.Tensor | None = None,
        norm_2: torch.Tensor | None = None,
    ) -> None:
        """Initialize the SOTA preprocessor.

        Args:
            dataset_name: Name of the dataset to load.
            n_wavelengths: Number of wavelengths.
            log_intensity: Whether to apply log intensity transformation.
            n_pca_comp: Number of PCA components.
            kfolds: Number of k-folds for cross-validation.
            fold: Current fold for cross-validation.
            norm_1: Pre-computed normalization mean.
            norm_2: Pre-computed normalization std.

        Raises:
            ValueError: If dataset_name is not supported.
            AssertionError: If n_wavelengths is not supported for the dataset.
        """
        self.dataset_name = dataset_name
        self.n_wavelengths = n_wavelengths
        self.log_intensity = log_intensity
        self.n_pca_comp = n_pca_comp
        self.kfolds = kfolds
        self.fold = fold

        # Initialize normalization parameters
        self.norm_1 = norm_1
        self.norm_2 = norm_2

        # Initialize PCA components and means
        self.pca_component_list: list = []
        self.pca_mean_list: list = []

        # Load and preprocess data
        self._load_data()
        self._preprocess_data()

    def _load_data(self) -> None:
        """Load data based on dataset name."""
        if self.dataset_name == "lan_lhs":
            self._load_lan_lhs_data()
        elif self.dataset_name == "lan":
            self._load_lan_data()
        elif self.dataset_name == "tsui":
            self._load_tsui_data()
        elif self.dataset_name == "manoj":
            self._load_manoj_data()
        else:
            raise ValueError(f"Dataset '{self.dataset_name}' not implemented")

    def _load_lan_lhs_data(self) -> None:
        """Load LAN LHS dataset."""
        if self.n_wavelengths != 1:
            raise ValueError("LAN LHS dataset only supports n_wavelengths=1")

        data = pd.read_parquet(os.path.join(BASE_DIR, "lan_lhs_2023_resim.parquet"))
        self.data = torch.from_numpy(
            data.drop(columns="layer [top first]").to_numpy()
        ).float()
        self.data = self.data.reshape(len(self.data), self.n_wavelengths, -1)

        self.params = self.data[..., :-1]
        self.labels = self.data[:, :, [-1]]

    def _load_lan_data(self) -> None:
        """Load LAN dataset."""
        if self.n_wavelengths != 1:
            raise ValueError("LAN dataset only supports n_wavelengths=1")

        data_lan = pd.read_parquet(os.path.join(BASE_DIR, "lan_2023_resim.parquet"))
        self.data = torch.from_numpy(
            data_lan.drop(columns="layer [top first]").to_numpy()
        ).float()
        self.data = self.data.reshape(len(self.data), self.n_wavelengths, -1)

        self.params = self.data[..., :-1]
        self.labels = self.data[:, :, [-1]]

    def _load_tsui_data(self) -> None:
        """Load TSUI dataset."""
        if self.n_wavelengths != 1:
            raise ValueError("TSUI dataset only supports n_wavelengths=1")

        data_tsui = pd.read_parquet(os.path.join(BASE_DIR, "tsui_2018_resim.parquet"))
        self.data = torch.from_numpy(
            data_tsui.drop(columns="layer [top first]").to_numpy()
        ).float()
        self.data = self.data.reshape(len(self.data), self.n_wavelengths, -1)

        self.params = self.data[..., :-1]
        self.labels = self.data[:, :, [-1]]

    def _load_manoj_data(self) -> None:
        """Load Manoj dataset."""
        if self.n_wavelengths != 351:
            raise ValueError("MANOJLOVIC dataset only supports n_wavelengths=351")

        loader = ManojlovicSimulationLoader(specular=False, thick_bottom=True)
        physio_params = loader.load_physiological_parameters()
        physical_params = loader.compute_physical_parameters(physio_params)
        reflectances = torch.from_numpy(loader.load_reflectances())

        # Use random subset of half the size for training (as in the paper)
        n_samples = len(reflectances) // 2
        torch.manual_seed(0)
        indices = torch.randperm(len(reflectances))[:n_samples]
        physical_params = physical_params[indices]
        reflectances = reflectances[indices]

        # Concatenate physical parameters and reflectances
        self.data = torch.cat(
            (physical_params, reflectances.unsqueeze(-1)), dim=-1
        ).float()
        self.data = self.data.reshape(len(self.data), self.n_wavelengths, -1)

        self.params = self.data[..., :-1]
        self.labels = self.data[:, :, [-1]]

    def _preprocess_data(self) -> None:
        """Preprocess the loaded data."""
        if self.dataset_name in ["lan_lhs", "lan"]:
            self._preprocess_lan_data()
        elif self.dataset_name == "tsui":
            self._preprocess_tsui_data()
        elif self.dataset_name == "manoj":
            self._preprocess_manoj_data()

        # Clamp reflectance to [0, 1]
        self.labels = torch.clamp(self.labels, min=0.0, max=1.0)

    def _preprocess_lan_data(self) -> None:
        """Preprocess LAN/LAN LHS data."""
        # Apply log10 to mu_a, mu_s, and d
        self.params[..., :2] = torch.log10(self.params[..., :2])
        self.params[..., -1] = torch.log10(self.params[..., -1])

    def _preprocess_tsui_data(self) -> None:
        """Preprocess TSUI data."""
        # Apply log10 to mu_a, mu_s, and d
        log_indices = [
            slice(0, 2),
            slice(5, 7),
            slice(10, 12),
            slice(15, 17),
            [4, 9, 14, 19],
        ]

        for idx in log_indices:
            self.params[..., idx] = torch.log10(self.params[..., idx])

    def _preprocess_manoj_data(self) -> None:
        """Preprocess Manoj data."""
        # Apply log10 to mu_a and mu_s
        self.params[..., :2] = torch.log10(self.params[..., :2])
        self.params[..., 5:7] = torch.log10(self.params[..., 5:7])

    def fit(self) -> torch.Tensor:
        """Fit the preprocessor and return processed data.

        Returns:
            Processed data tensor.
        """
        # Collect training data
        train_ids = self.consistent_data_split_ids(self.params, "train")

        # Apply z-score normalization (compute only on training data, apply to all data)
        logger.info("Applying z-score normalization")
        logger.info(f"Training data shape: {self.params[train_ids].shape}")

        if self.params.ndim == 2:
            self.norm_1 = self.params[train_ids].mean(dim=0)
            self.norm_2 = self.params[train_ids].std(dim=0)
        else:
            self.norm_1 = self.params[train_ids].mean(dim=(0, 1))
            self.norm_2 = self.params[train_ids].std(dim=(0, 1))

        logger.info(f"Mean shape: {self.norm_1.shape}")
        logger.info(f"Std shape: {self.norm_2.shape}")

        parameters = (self.params - self.norm_1) / torch.clamp(
            self.norm_2, min=1e-8, max=None
        )

        # Sanity checking
        logger.info("Mean and std after applying normalization")
        if parameters.ndim == 2:
            logger.info(f"Mean: {parameters.mean(dim=0)}")
            logger.info(f"Std: {parameters.std(dim=0)}")
        else:
            logger.info(f"Mean: {parameters.mean(dim=(0, 1))}")
            logger.info(f"Std: {parameters.std(dim=(0, 1))}")

        logger.info(f"Parameters shape: {parameters.shape}")

        return torch.cat((self.params, self.labels), dim=-1)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply previously initialized parameter pre-processing steps.

        Args:
            data: Input data tensor of shape (n_samples, n_wvl, n_param + 1).

        Returns:
            Processed data tensor.

        Raises:
            ValueError: If data shape is incorrect.
        """
        if data.ndim != 3:
            raise ValueError(f"Data must be 3D tensor, got shape {data.shape}")

        params, reflectance = data[..., :-1], data[..., [-1]]
        params = (params - self.norm_1) / torch.clamp(self.norm_2, min=1e-8, max=None)
        reflectance = torch.clamp(reflectance, min=0.0, max=1.0)

        return torch.cat((params, reflectance), dim=-1)

    def consistent_data_split_ids(
        self,
        raw_data: torch.Tensor,
        mode: str,
        kfolds: int | None = None,
        fold: int | None = None,
    ) -> np.ndarray:
        """Split data consistently into train, validation, and test sets.

        Args:
            raw_data: Input data.
            mode: Split mode ('train', 'val', 'test').
            kfolds: Number of k-folds (not used in this implementation).
            fold: Current fold (not used in this implementation).

        Returns:
            Array of indices for the specified split.

        Raises:
            ValueError: If mode is not supported.
        """
        if mode not in ["train", "val", "test"]:
            raise ValueError(f"Mode must be 'train', 'val', or 'test', got {mode}")

        # Split data: 70% train, 10% val, 20% test
        train_ratio, val_ratio, _ = 0.7, 0.1, 0.2
        n_samples = len(raw_data)

        train_idx = int(n_samples * train_ratio)
        val_idx = int(n_samples * (train_ratio + val_ratio))

        if mode == "train":
            return np.arange(train_idx)
        elif mode == "val":
            return np.arange(train_idx, val_idx)
        else:
            return np.arange(val_idx, n_samples)


class SOTADataset(Dataset):
    """Simplified Dataset for SOTA datasets."""

    def __init__(
        self,
        data: torch.Tensor,
        n_wavelengths: int,
        n_params: int,
        thick_deepest_layer: bool = True,
        mix_data: Any = None,
        preprocessor: Any = None,
    ) -> None:
        """Initialize the SOTADataset.

        Args:
            data: The data to be used.
            n_wavelengths: The number of wavelengths.
            n_params: The number of parameters.
            thick_deepest_layer: Whether the deepest layer is thick.
            mix_data: Whether to mix data (dummy for compatibility).
            preprocessor: The preprocessor to be used (dummy for compatibility).

        Raises:
            ValueError: If data shape is incorrect.
        """
        super().__init__()

        if data.ndim != 3:
            raise ValueError(f"Data must be 3D tensor, got shape {data.shape}")

        self.data = data
        self.n_wvl = n_wavelengths
        self.n_params = n_params
        self.thick_deepest_layer = thick_deepest_layer

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return int(self.data.shape[0])

    def __getitem__(self, index: int | list[int] | torch.Tensor) -> torch.Tensor:
        """Load the raw data and preprocess it on the fly in collate_fn."""
        return self.data[index]
