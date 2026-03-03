"""Clamped dataset data loading functionality."""

import os
from pathlib import Path

import pandas as pd
from htc import DatasetImage
from htc.models.data.DataSpecification import DataSpecification
from htc.utils.Config import Config
from torch.utils.data import DataLoader

from mcmlnet.experiments.data_loaders.config import DataConfig


class ClampingDataLoader:
    """Loader for clamping datasets, based on different data specifications."""

    def __init__(
        self,
        data_specification_name: str = "clinical_application_pig_aortic_single_fold.json",
        annotation_name: str = "semantic#primary",
    ) -> None:
        """Initialize combined human data loader."""
        self.fold = "fold_0"
        self.data_specification_name = data_specification_name
        self.annotation_name = annotation_name

        # Define metadata file name
        metadata_file_names = {
            "clinical_application_pig_aortic_single_fold.json": "df_aortic_pig",
        }
        try:
            self.metadata_name = metadata_file_names[self.data_specification_name]
        except KeyError as err:
            raise ValueError(
                f"Unknown data specification: {self.data_specification_name}"
            ) from err

    def _get_data_specification(self) -> DataSpecification:
        """Get the appropriate data specification files.

        Returns:
            Tuple of (semantic_spec, polygon_spec).
        """
        return DataSpecification(
            Path(os.environ["cache_dir"]) / self.data_specification_name
        )

    def _define_dataloader(self, split: str) -> DataLoader:
        """Define dataloader for indidividual splits.

        Args:
            split: Data split name to load from folds.

        Returns:
            DataLoader
        """
        if split not in ["val", "test"]:
            raise ValueError(f"split should be either 'val' or 'test' got: {split}.")

        spec = self._get_data_specification()
        fold = spec.folds[self.fold]

        if split == "val":
            paths = fold[split]
        else:
            with spec.activated_test_set():
                paths = spec.paths("^test")

        # Define configuration
        config = Config(
            {
                "input/n_channels": DataConfig.DEFAULT_N_WAVELENGTHS,
                "input/annotation_name": self.annotation_name,
            }
        )

        # Define Dataset and DataLoader
        dataset = DatasetImage(paths, train=False, config=config)

        return DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

    def _load_metadata(self, split: str) -> pd.DataFrame:
        """Load metadata tables for a specific split.

        Args:
            split: Data split name.

        Returns:
            DataFrame containing metadata
        """
        return pd.read_csv(
            os.path.join(os.environ["cache_dir"], f"{self.metadata_name}_{split}.csv")
        )

    def load_validation_data(self) -> tuple[DataLoader, pd.DataFrame]:
        """Load validation data and metadata for initialized data specification.

        Returns:
            Tuple of validation DataLoader and validation metadata DataFrame.
        """
        return self._define_dataloader("val"), self._load_metadata("val")

    def load_test_data(self) -> tuple[DataLoader, pd.DataFrame]:
        """Load test data and metadata for initialized data specification.

        Returns:
            Tuple of test DataLoader and test metadata DataFrame.
        """
        return self._define_dataloader("test"), self._load_metadata("test")
