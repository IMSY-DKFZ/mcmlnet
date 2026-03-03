"""
Common and configurable PyTorch Lightning DataModule.
"""

from copy import deepcopy
from functools import partial
from typing import Any

import lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader

from mcmlnet.training.data_loading.preprocessing import (
    PreProcessor,
    collate_variable_tensors,
)
from mcmlnet.utils.logging import setup_logging

logger = setup_logging(level="info", logger_name=__name__)


class DataModule(pl.LightningDataModule):
    """Data module for MC simulation data."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the data module.

        Args:
            cfg: Configuration dictionary.

        Raises:
            ValueError: If configuration parameters are invalid.
        """
        super().__init__()

        # Validate configuration
        self._validate_config(cfg)

        self.n_workers = cfg.n_workers
        self.batch_size = cfg.batch_size
        self.max_augm_epochs = cfg.max_augm_epochs

        # Initialize preprocessor and fit it
        self.preprocessor = instantiate(cfg.preprocessing)
        raw_data = self.preprocessor.fit()

        # Store normalization in config
        self._store_normalization_in_config(cfg)
        self.cfg = cfg

        # Split data and create datasets
        self._create_datasets(raw_data, cfg)

        # Handle additional scaling datasets if specified
        if cfg.get("additional_scaling"):
            self._add_scaling_datasets(cfg)

    def _validate_config(self, cfg: DictConfig) -> None:
        """Validate configuration parameters."""
        required_params = {
            "n_workers": (
                "non-negative integer",
                lambda x: isinstance(x, int) and x >= 0,
            ),
            "batch_size": ("positive integer", lambda x: isinstance(x, int) and x > 0),
            "max_augm_epochs": (
                "non-negative integer",
                lambda x: isinstance(x, int) and x >= 0,
            ),
        }

        for param, (description, validator) in required_params.items():
            if not hasattr(cfg, param) or not validator(getattr(cfg, param)):
                raise ValueError(
                    f"{param} must be {description}, "
                    f"got {getattr(cfg, param, 'missing')}"
                )

    def _store_normalization_in_config(self, cfg: DictConfig) -> None:
        """Store normalization parameters in config."""
        with open_dict(cfg):
            if self.preprocessor.norm_1 is not None:
                cfg.preprocessing.norm_1 = self.preprocessor.norm_1.tolist()
            if self.preprocessor.norm_2 is not None:
                cfg.preprocessing.norm_2 = self.preprocessor.norm_2.tolist()
            if (
                isinstance(self.preprocessor.pca_component_list, list)
                and len(self.preprocessor.pca_component_list) != 0
            ):
                cfg.preprocessing.pca_transformation_list = (
                    self.preprocessor.pca_component_list
                )
                cfg.preprocessing.pca_mean_list = self.preprocessor.pca_mean_list

        # Store config
        self.cfg = cfg

    def _create_dataset(
        self,
        dataset_cfg: DictConfig,
        data: torch.Tensor,
        preprocessor: PreProcessor,
        is_valid_or_test: bool = False,
    ) -> Any:
        """Create a dataset with the given configuration."""
        return instantiate(
            dataset_cfg,
            data=data,
            preprocessor=preprocessor,
            mix_data=False if is_valid_or_test else True,
        )

    def _apply_training_limitation(
        self, train_data: torch.Tensor, cfg: DictConfig
    ) -> torch.Tensor:
        """Apply artificial training data limitation."""
        train_data_ratio = cfg.get("train_data_ratio")
        if train_data_ratio not in [None, "1", "1.0", 1]:
            seed = cfg.preprocessing.get("fold", 0)
            shuffled_ids = torch.randperm(
                len(train_data), generator=torch.Generator().manual_seed(seed)
            )
            train_data = train_data[shuffled_ids]
            train_data = train_data[: int(len(train_data) * train_data_ratio)]
        return train_data

    def _create_datasets(self, raw_data: torch.Tensor, cfg: DictConfig) -> None:
        """Create train, validation, and test datasets."""
        # Split data
        self.train_ids = self.preprocessor.consistent_data_split_ids(
            raw_data,
            mode="train",
            kfolds=cfg.preprocessing.kfolds,
            fold=cfg.preprocessing.fold,
        )
        self.val_ids = self.preprocessor.consistent_data_split_ids(raw_data, mode="val")
        self.test_ids = self.preprocessor.consistent_data_split_ids(
            raw_data, mode="test"
        )

        # Apply training data limitation
        train_data = self._apply_training_limitation(raw_data[self.train_ids], cfg)

        # Create datasets
        self.train_ = self._create_dataset(cfg.dataset, train_data, self.preprocessor)
        self.val_ = self._create_dataset(
            cfg.dataset,
            raw_data[self.val_ids],
            self.preprocessor,
            is_valid_or_test=True,
        )
        self.test_ = self._create_dataset(
            cfg.dataset,
            raw_data[self.test_ids],
            self.preprocessor,
            is_valid_or_test=True,
        )

    def _add_scaling_datasets(self, cfg: DictConfig) -> None:
        """Add additional scaling datasets."""
        concat_sets = cfg.get("additional_scaling")
        if not concat_sets:
            return

        # Load additional scaling dataset
        dataset_loader_scaling = instantiate(concat_sets["dataset_loader"])
        scaling_data = dataset_loader_scaling.load_data(
            concat_sets["data"], concat_sets["n_wavelengths"]
        )
        if concat_sets.get("use_only_3M_data"):
            first_half = len(scaling_data) // 2
            scaling_data = scaling_data[:first_half]

        # Split scaling data
        scaling_train_ids = self.preprocessor.consistent_data_split_ids(
            scaling_data,
            mode="train",
        )
        scaling_val_ids = self.preprocessor.consistent_data_split_ids(
            scaling_data, mode="val"
        )
        scaling_test_ids = self.preprocessor.consistent_data_split_ids(
            scaling_data, mode="test"
        )

        # Apply training limitation to scaling data
        scaling_train_data = self._apply_training_limitation(
            scaling_data[scaling_train_ids], cfg
        )

        # Create copied preprocessor for scaling datasets - update is_physical flag
        copied_preprocessor = deepcopy(self.preprocessor)
        copied_preprocessor.data_loader.is_physical = concat_sets["dataset_loader"][
            "is_physical"
        ]

        # Create scaling datasets
        scaling_train = self._create_dataset(
            cfg.dataset, scaling_train_data, copied_preprocessor
        )
        scaling_val = self._create_dataset(
            cfg.dataset,
            scaling_data[scaling_val_ids],
            copied_preprocessor,
            is_valid_or_test=True,
        )
        scaling_test = self._create_dataset(
            cfg.dataset,
            scaling_data[scaling_test_ids],
            copied_preprocessor,
            is_valid_or_test=True,
        )

        # Concatenate datasets
        self.train_ = torch.utils.data.ConcatDataset([self.train_, scaling_train])
        self.val_ = torch.utils.data.ConcatDataset([self.val_, scaling_val])
        self.test_ = torch.utils.data.ConcatDataset([self.test_, scaling_test])

    def _create_dataloader(
        self,
        dataset: Any,
        batch_size: int | None = None,
        shuffle: bool = False,
        num_workers: int = 4,
    ) -> DataLoader:
        """Create a data loader with common settings."""
        return DataLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=partial(
                collate_variable_tensors,
                preprocessor=self.preprocessor,
                thick_deepest_layer=self.cfg.dataset.thick_deepest_layer,
                n_params=self.cfg.dataset.n_params,
            ),
            pin_memory=False,
        )

    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        logger.info(
            "Creating/ updating train DataLoader... Dataset length: %d",
            len(self.train_),
        )
        return self._create_dataloader(
            self.train_, shuffle=True, num_workers=self.n_workers
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation data loader."""
        return self._create_dataloader(self.val_, shuffle=False, batch_size=512)

    def test_dataloader(self) -> DataLoader:
        """Create test data loader."""
        return self._create_dataloader(self.test_, shuffle=False, batch_size=512)
