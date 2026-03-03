"""Real data loading functionality."""

import os
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from htc import DatasetImage
from htc.models.data.DataSpecification import DataSpecification
from htc.utils.Config import Config
from torch.utils.data import DataLoader

from mcmlnet.experiments.data_loaders.config import DataConfig
from mcmlnet.utils.logging import setup_logging

logger = setup_logging(level="info", logger_name=__name__)


class RealDataLoader(ABC):
    """Abstract base class for real data loaders."""

    def __init__(self, dataset_name: str):
        """Initialize real data loader.

        Args:
            dataset_name: Name of the dataset.
        """
        self.dataset_name = dataset_name

    @abstractmethod
    def load_training_data(self) -> tuple[DataLoader, DataLoader]:
        """Load training and validation data."""

    @abstractmethod
    def load_test_data(self) -> DataLoader:
        """Load test data."""

    @abstractmethod
    def load_final_data(self) -> DataLoader:
        """Load final data."""

    @abstractmethod
    def load_data(self) -> dict[str, DataLoader]:
        """Load all data."""


class PigDataLoader(RealDataLoader):
    """Loader for pig dataset data."""

    def __init__(self, dataset_type: str = "semantic"):
        """Initialize pig data loader.

        Args:
            dataset_type: Type of pig dataset ('semantic' or 'masks').
        """
        super().__init__("pig")
        self.dataset_type = dataset_type

        if dataset_type not in ["semantic", "masks"]:
            raise ValueError(f"Unsupported pig dataset type: {dataset_type}")

        self.fold = "fold_0" if dataset_type == "masks" else "fold_P044,P050,P059"

    def _get_data_specification(self) -> str:
        """Get the appropriate data specification file.

        Returns:
            Data specification filename.
        """
        if self.dataset_type == "semantic":
            return "pigs_semantic-only_5foldsV2.json"
        else:  # masks
            return "pigs_masks_fold-baseline_4cam.json"

    def _get_split_names(self) -> tuple[str, str]:
        """Get training and validation split names.

        Returns:
            Tuple of (train_split, val_split) names.
        """
        if self.dataset_type == "semantic":
            return "train_semantic", "val_semantic_unknown"
        else:  # masks
            return "train", "val"

    def load_training_data(self) -> tuple[DataLoader, DataLoader]:
        """Load training and validation data for pig dataset.

        Returns:
            Tuple of (training_dataloader, validation_dataloader).
        """
        data_spec = self._get_data_specification()
        train_split, val_split = self._get_split_names()

        # Define configuration
        config = Config({"input/n_channels": DataConfig.DEFAULT_N_WAVELENGTHS})
        specs = DataSpecification(data_spec)
        fold = specs.folds[self.fold]

        # Create training dataloader
        train_dataset = DatasetImage(fold[train_split], train=False, config=config)
        train_dataloader = DataLoader(
            train_dataset, batch_size=1, num_workers=0, shuffle=False
        )
        logger.info(f"Training samples: {len(train_dataset)}")

        # Create validation dataloader
        val_dataset = DatasetImage(fold[val_split], train=False, config=config)
        val_dataloader = DataLoader(
            val_dataset, batch_size=1, num_workers=0, shuffle=False
        )
        logger.info(f"Validation samples: {len(val_dataset)}")

        return train_dataloader, val_dataloader

    def load_test_data(self) -> DataLoader:
        """Load test data for pig dataset.

        Returns:
            Test dataloader.
        """
        data_spec = self._get_data_specification()

        # Define configuration
        config = Config({"input/n_channels": DataConfig.DEFAULT_N_WAVELENGTHS})
        specs = DataSpecification(data_spec)

        # Get test paths
        with specs.activated_test_set():
            test_paths = specs.paths("^test")

        # Create test dataloader
        test_dataset = DatasetImage(test_paths, train=False, config=config)
        test_dataloader = DataLoader(
            test_dataset, batch_size=1, num_workers=0, shuffle=False
        )
        logger.info(f"Test samples: {len(test_dataset)}")

        return test_dataloader

    def load_final_data(self) -> DataLoader:
        """Load pig dataset (train + val + test).

        Returns:
            Final dataloader containing all data.
        """
        # Load individual splits
        train_dl, val_dl = self.load_training_data()
        test_dl = self.load_test_data()

        # Combine all datasets
        train_dataset = train_dl.dataset
        val_dataset = val_dl.dataset
        test_dataset = test_dl.dataset

        final_dataset = torch.utils.data.ConcatDataset(
            [train_dataset, val_dataset, test_dataset]
        )

        logger.info(f"Final dataset size: {len(final_dataset)}")

        # Create final dataloader
        final_dataloader = DataLoader(
            final_dataset, batch_size=1, num_workers=0, shuffle=False
        )

        return final_dataloader

    def load_data(self) -> dict[str, DataLoader]:
        """Load all pig dataset data.

        Returns:
            Dictionary containing training, validation, and test dataloaders.
        """
        train_dl, val_dl = self.load_training_data()
        test_dl = self.load_test_data()

        return {
            "train": train_dl,
            "val": val_dl,
            "test": test_dl,
        }


class HumanDataLoader(RealDataLoader):
    """Loader for human dataset data."""

    def __init__(self) -> None:
        """Initialize human data loader."""
        super().__init__("human")
        self.fold = "fold_0"

    def _get_data_specification(self) -> str:
        """Get the appropriate data specification file.

        Returns:
            Data specification filename.
        """
        return "human_semantic-only_physiological-kidney_5folds_nested-0-2_mapping-12_seed-0.json"  # noqa: E501

    def load_training_data(self) -> tuple[DataLoader, DataLoader]:
        """Load training and validation data for human dataset.

        Returns:
            Tuple of (training_dataloader, validation_dataloader).
        """
        data_spec = self._get_data_specification()

        # Define configuration
        config = Config({"input/n_channels": DataConfig.DEFAULT_N_WAVELENGTHS})
        specs = DataSpecification(data_spec)
        fold = specs.folds[self.fold]

        # Create training dataloader
        train_dataset = DatasetImage(fold["train"], train=False, config=config)
        train_dataloader = DataLoader(
            train_dataset, batch_size=1, num_workers=0, shuffle=False
        )
        logger.info(f"Training samples: {len(train_dataset)}")

        # Create validation dataloader
        val_dataset = DatasetImage(fold["val"], train=False, config=config)
        val_dataloader = DataLoader(
            val_dataset, batch_size=1, num_workers=0, shuffle=False
        )
        logger.info(f"Validation samples: {len(val_dataset)}")

        return train_dataloader, val_dataloader

    def load_test_data(self) -> DataLoader:
        """Load test data for human dataset.

        Returns:
            Test dataloader.
        """
        data_spec = self._get_data_specification()

        # Define configuration
        config = Config({"input/n_channels": DataConfig.DEFAULT_N_WAVELENGTHS})
        specs = DataSpecification(data_spec)

        # Get test paths
        with specs.activated_test_set():
            test_paths = specs.paths("^test")

        # Create test dataloader
        test_dataset = DatasetImage(test_paths, train=False, config=config)
        test_dataloader = DataLoader(
            test_dataset, batch_size=1, num_workers=0, shuffle=False
        )
        logger.info(f"Test samples: {len(test_dataset)}")

        return test_dataloader

    def load_final_data(self) -> DataLoader:
        """Load human dataset (train + val + test).

        Returns:
            Final dataloader containing all data.
        """
        # Load individual splits
        train_dl, val_dl = self.load_training_data()
        test_dl = self.load_test_data()

        # Combine all datasets
        train_dataset = train_dl.dataset
        val_dataset = val_dl.dataset
        test_dataset = test_dl.dataset

        final_dataset = torch.utils.data.ConcatDataset(
            [train_dataset, val_dataset, test_dataset]
        )

        logger.info(f"Final dataset size: {len(final_dataset)}")

        # Create final dataloader
        final_dataloader = DataLoader(
            final_dataset, batch_size=1, num_workers=0, shuffle=False
        )

        return final_dataloader

    def load_data(self) -> dict[str, DataLoader]:
        """Load all human dataset data.

        Returns:
            Dictionary containing training, validation, and test dataloaders.
        """
        train_dl, val_dl = self.load_training_data()
        test_dl = self.load_test_data()

        return {
            "train": train_dl,
            "val": val_dl,
            "test": test_dl,
        }


class CombinedHumanDataLoader(RealDataLoader):
    """Loader for combined human dataset (semantic + polygon)."""

    def __init__(self) -> None:
        """Initialize combined human data loader."""
        super().__init__("human_combined")
        self.fold = "fold_0"

    def _get_data_specifications(self) -> tuple[DataSpecification, DataSpecification]:
        """Get the appropriate data specification files.

        Returns:
            Tuple of (semantic_spec, polygon_spec).
        """
        specs_semantic = DataSpecification(
            "human_semantic-only_physiological-kidney_5folds_nested-0-2_mapping-12_seed-0.json"
        )
        specs_polygon = DataSpecification(
            Path(os.environ["cache_dir"])
            / "human_masks-only_physiological-kidney_5folds_nested-0-2_mapping-12_seed-0.json"  # noqa: E501
        )
        return specs_semantic, specs_polygon

    def load_training_data(self) -> tuple[DataLoader, DataLoader]:
        """Load training and validation data for combined human dataset.

        Returns:
            Tuple of (training_dataloader, validation_dataloader).
        """
        # Load data specifications
        semantic_spec, polygon_spec = self._get_data_specifications()
        fold_semantic = semantic_spec.folds[self.fold]
        fold_polygon = polygon_spec.folds[self.fold]

        # Define configuration
        config = Config({"input/n_channels": DataConfig.DEFAULT_N_WAVELENGTHS})

        # Load semantic data
        train_dataset_semantic = DatasetImage(
            fold_semantic["train"], train=False, config=config
        )
        val_dataset_semantic = DatasetImage(
            fold_semantic["val"], train=False, config=config
        )

        logger.info(f"Training images semantic: {len(train_dataset_semantic)}")
        logger.info(f"Validation images semantic: {len(val_dataset_semantic)}")

        # Load polygon data
        train_dataset_polygon = DatasetImage(
            fold_polygon["train"], train=False, config=config
        )
        val_dataset_polygon = DatasetImage(
            fold_polygon["val"], train=False, config=config
        )

        logger.info(f"Training images polygon: {len(train_dataset_polygon)}")
        logger.info(f"Validation images polygon: {len(val_dataset_polygon)}")

        # Combine datasets
        train_dataset = torch.utils.data.ConcatDataset(
            [train_dataset_semantic, train_dataset_polygon]
        )
        val_dataset = torch.utils.data.ConcatDataset(
            [val_dataset_semantic, val_dataset_polygon]
        )

        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset, batch_size=1, num_workers=0, shuffle=False
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=1, num_workers=0, shuffle=False
        )

        return train_dataloader, val_dataloader

    def load_test_data(self) -> DataLoader:
        """Load test data for combined human dataset.

        Returns:
            Test dataloader.
        """
        semantic_spec, polygon_spec = self._get_data_specifications()

        # Define configuration
        config = Config({"input/n_channels": DataConfig.DEFAULT_N_WAVELENGTHS})

        # Load semantic test data
        with semantic_spec.activated_test_set():
            test_paths_semantic = semantic_spec.paths("^test")
        test_dataset_semantic = DatasetImage(
            test_paths_semantic, train=False, config=config
        )

        # Load polygon test data
        with polygon_spec.activated_test_set():
            test_paths_polygon = polygon_spec.paths("^test")
        test_dataset_polygon = DatasetImage(
            test_paths_polygon, train=False, config=config
        )

        logger.info(f"Test images semantic: {len(test_dataset_semantic)}")
        logger.info(f"Test images polygon: {len(test_dataset_polygon)}")

        # Combine test datasets
        test_dataset = torch.utils.data.ConcatDataset(
            [test_dataset_semantic, test_dataset_polygon]
        )

        # Create test dataloader
        test_dataloader = DataLoader(
            test_dataset, batch_size=1, num_workers=0, shuffle=False
        )

        return test_dataloader

    def load_final_data(self) -> DataLoader:
        """Load final combined human dataset (train + val + test).

        Returns:
            Final dataloader containing all data.
        """
        # Load individual splits
        train_dl, val_dl = self.load_training_data()
        test_dl = self.load_test_data()

        # Combine all datasets
        train_dataset = train_dl.dataset
        val_dataset = val_dl.dataset
        test_dataset = test_dl.dataset

        final_dataset = torch.utils.data.ConcatDataset(
            [train_dataset, val_dataset, test_dataset]
        )

        logger.info(f"Final dataset size: {len(final_dataset)}")

        # Create final dataloader
        final_dataloader = DataLoader(
            final_dataset, batch_size=1, num_workers=0, shuffle=False
        )

        return final_dataloader

    def load_data(self) -> dict[str, DataLoader]:
        """Load all combined human dataset data.

        Returns:
            Dictionary containing training, validation, test, and final dataloaders.
        """
        train_dl, val_dl = self.load_training_data()
        test_dl = self.load_test_data()

        return {
            "train": train_dl,
            "val": val_dl,
            "test": test_dl,
        }
