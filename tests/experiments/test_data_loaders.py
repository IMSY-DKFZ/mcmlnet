"""Tests for mcmlnet.experiments.data_loaders modules."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from mcmlnet.experiments.data_loaders.aggregation import (
    aggregate_data_image_level,
    aggregate_data_subject_level,
)
from mcmlnet.experiments.data_loaders.config import DataConfig
from mcmlnet.experiments.data_loaders.real_data import (
    CombinedHumanDataLoader,
    HumanDataLoader,
)
from mcmlnet.experiments.data_loaders.simulation import SimulationDataLoaderManager
from mcmlnet.experiments.data_loaders.utils import (
    create_data_summary,
    get_image_id,
    get_subject_id,
    subsample_data,
)


class DummyDictDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.features = torch.randn(10, 100, 10)
        self.labels = torch.randint(0, 5, (10, 100))
        self.image_names = [f"{i:012d}_test" for i in range(10)]

    def __len__(self) -> int:
        return 10

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        return {
            "features": self.features[idx],
            "labels": self.labels[idx],
            "image_name": self.image_names[idx],
        }


@pytest.fixture  # type: ignore[misc]
def dummy_dataloader() -> DataLoader:
    """Fixture for a dummy dataloader yielding dicts."""
    return DataLoader(DummyDictDataset(), batch_size=1)


class TestUtils:
    """Test cases for utils module."""

    @pytest.mark.parametrize(  # type: ignore[misc]
        "data,data_type",
        [(np.random.randn(100, 10), "numpy"), (torch.randn(100, 10), "torch")],
    )
    def test_subsample_data_valid(
        self, data: np.ndarray | torch.Tensor, data_type: str
    ) -> None:
        """Test valid data subsampling."""
        subsampled = subsample_data(data, 50, random_seed=42)

        if data_type == "torch":
            assert isinstance(subsampled, torch.Tensor)
        else:
            assert isinstance(subsampled, np.ndarray)
        assert subsampled.shape == (50, 10)

    @pytest.mark.parametrize(  # type: ignore[misc]
        "data,data_type",
        [(np.random.randn(50, 10), "numpy"), (torch.randn(50, 10), "torch")],
    )
    def test_subsample_data_smaller_size(
        self, data: np.ndarray | torch.Tensor, data_type: str
    ) -> None:
        """Test subsampling when target size is smaller than data size."""
        subsampled = subsample_data(data, 100, random_seed=42)

        if data_type == "torch":
            assert isinstance(subsampled, torch.Tensor)
            assert torch.equal(subsampled, data)
        else:
            assert isinstance(subsampled, np.ndarray)
            assert np.array_equal(subsampled, data)
        assert subsampled.shape == (50, 10)

    def test_subsample_data_invalid_size(self) -> None:
        """Test error with invalid subsample size."""
        data = np.random.randn(100, 10)

        with pytest.raises(ValueError, match="Subsample size must be positive"):
            subsample_data(data, 0)

        with pytest.raises(ValueError, match="Subsample size must be positive"):
            subsample_data(data, -10)

    @pytest.mark.parametrize(  # type: ignore[misc]
        "dataset",
        ["pig_semantic", "pig_masks", "human"],
    )
    def test_get_subject_id(self, dataset: str) -> None:
        """Test subject ID extraction for different datasets."""
        subject_id = get_subject_id("1234_5678_90.excluded", dataset)

        isinstance(subject_id, str)
        if dataset in ["pig_semantic", "pig_masks"]:
            assert subject_id == "1234"
        else:
            assert subject_id == "1234_5678_90"

    def test_get_subject_id_unsupported_dataset(self) -> None:
        """Test error with unsupported dataset type."""
        with pytest.raises(ValueError, match="Unsupported dataset type"):
            get_subject_id("1234_5678_90.excluded", "unsupported")

    @pytest.mark.parametrize(  # type: ignore[misc]
        "dataset",
        ["pig_semantic", "pig_masks", "human"],
    )
    def test_get_image_id_pig_semantic(self, dataset: str) -> None:
        """Test image ID extraction for different datasets."""
        image_id = get_image_id("excl.1234_5678", dataset)

        isinstance(image_id, str)
        if dataset in ["pig_semantic", "pig_masks"]:
            assert image_id == "1234_5678"
        else:
            assert image_id == "8"

    def test_get_image_id_unsupported_dataset(self) -> None:
        """Test error with unsupported dataset type."""
        with pytest.raises(ValueError, match="Unsupported dataset type"):
            get_image_id("excl.1234_5678", "unsupported")

    @pytest.mark.parametrize(  # type: ignore[misc]
        "data,labels",
        [
            (np.random.randn(100, 10), np.random.randint(0, 5, 100)),
            (torch.randn(100, 10), torch.randint(0, 5, (100,))),
        ],
    )
    def test_create_data_summary(
        self, data: np.ndarray | torch.Tensor, labels: np.ndarray | torch.Tensor
    ) -> None:
        """Test data summary creation."""
        subject_ids = [f"subject_{i}" for i in range(len(data))]

        for labels_option in [labels, None]:
            for subject_ids_option in [subject_ids, None]:
                summary = create_data_summary(data, labels_option, subject_ids_option)

                assert "data_shape" in summary
                assert "data_type" in summary
                assert "data_range" in summary
                assert "data_mean" in summary
                assert "data_std" in summary
                assert summary["data_shape"] == (100, 10)

                if labels_option is not None:
                    assert "unique_labels" in summary
                    assert summary["unique_labels"] == 5  # 0-4

                if subject_ids_option is not None:
                    assert "unique_subjects" in summary
                    assert summary["unique_subjects"] == 100


class TestAggregation:
    """Test cases for aggregation module."""

    @pytest.mark.parametrize(  # type: ignore[misc]
        "fit_on_organic_only", [True, False]
    )
    def test_aggregate_data_image_level_mock_dataloader(
        self, dummy_dataloader: DataLoader, fit_on_organic_only: bool
    ) -> None:
        """Test image-level data aggregation with valid dataloader."""
        result = aggregate_data_image_level(
            dummy_dataloader, "human", fit_on_organic_only=fit_on_organic_only
        )

        assert "mean_spectra" in result
        assert "label_ids" in result
        assert "subject_ids" in result
        assert "image_ids" in result
        assert isinstance(result["mean_spectra"], np.ndarray)
        assert isinstance(result["label_ids"], list)
        assert isinstance(result["subject_ids"], list)
        assert isinstance(result["image_ids"], list)

        assert len(np.unique(result["subject_ids"])) == 10
        assert len(np.unique(result["image_ids"])) == 1

        if fit_on_organic_only:
            assert result["mean_spectra"].shape == (30, 10)
            assert len(result["label_ids"]) == 30
            assert len(result["subject_ids"]) == 30
            assert len(result["image_ids"]) == 30

            assert len(np.unique(result["mean_spectra"])) == 300
            assert len(np.unique(result["label_ids"])) == 3
        else:
            assert result["mean_spectra"].shape == (50, 10)
            assert len(result["label_ids"]) == 50
            assert len(result["subject_ids"]) == 50
            assert len(result["image_ids"]) == 50

            assert len(np.unique(result["mean_spectra"])) == 500
            assert len(np.unique(result["label_ids"])) == 5

    @pytest.mark.parametrize(  # type: ignore[misc]
        "fit_on_organic_only", [True, False]
    )
    def test_aggregate_data_image_level_invalid_batch_size(
        self, fit_on_organic_only: bool
    ) -> None:
        """Test error with invalid batch size."""
        dataloader = DataLoader(DummyDictDataset(), batch_size=2)

        with pytest.raises(ValueError, match="Batch size must be 1"):
            aggregate_data_image_level(
                dataloader, "human", fit_on_organic_only=fit_on_organic_only
            )

    @pytest.mark.parametrize(  # type: ignore[misc]
        "fit_on_organic_only", [True, False]
    )
    def test_aggregate_data_subject_level_mock_dataloader(
        self, dummy_dataloader: DataLoader, fit_on_organic_only: bool
    ) -> None:
        """Test subject-level data aggregation with mock dataloader."""

        result = aggregate_data_subject_level(
            dummy_dataloader, "human", fit_on_organic_only=fit_on_organic_only
        )

        assert "mean_spectra" in result
        assert "label_ids" in result
        assert "subject_ids" in result
        assert isinstance(result["mean_spectra"], np.ndarray)
        assert isinstance(result["label_ids"], list)
        assert isinstance(result["subject_ids"], list)

        assert len(np.unique(result["subject_ids"])) == 10

        if fit_on_organic_only:
            assert result["mean_spectra"].shape == (30, 10)
            assert len(result["label_ids"]) == 30
            assert len(result["subject_ids"]) == 30

            assert len(np.unique(result["mean_spectra"])) == 300
            assert len(np.unique(result["label_ids"])) == 3
        else:
            assert result["mean_spectra"].shape == (50, 10)
            assert len(result["label_ids"]) == 50
            assert len(result["subject_ids"]) == 50

            assert len(np.unique(result["mean_spectra"])) == 500
            assert len(np.unique(result["label_ids"])) == 5

    @pytest.mark.parametrize(  # type: ignore[misc]
        "fit_on_organic_only", [True, False]
    )
    def test_aggregate_data_subject_level_invalid_batch_size(
        self, fit_on_organic_only: bool
    ) -> None:
        """Test error with invalid batch size."""
        dataloader = DataLoader(DummyDictDataset(), batch_size=2)

        with pytest.raises(ValueError, match="Batch size must be 1"):
            aggregate_data_subject_level(
                dataloader, "human", fit_on_organic_only=fit_on_organic_only
            )


class TestConfig:
    """Test cases for config module."""

    def test_data_config_constants(self) -> None:
        """Test DataConfig constants."""
        config = DataConfig()

        # Test supported simulations
        assert "generic_sims" in config.SUPPORTED_SIMULATIONS
        assert "lan_sims" in config.SUPPORTED_SIMULATIONS
        assert "tsui_sims" in config.SUPPORTED_SIMULATIONS

        # Test supported ablations
        assert "base_subset" in config.SUPPORTED_ABLATIONS
        assert "superset" in config.SUPPORTED_ABLATIONS

        # Test supported datasets
        assert "pig_semantic" in config.SUPPORTED_DATASETS
        assert "human" in config.SUPPORTED_DATASETS
        assert "pig_masks" in config.SUPPORTED_DATASETS

    def test_simulation_names(self) -> None:
        """Test simulation name mappings."""
        config = DataConfig()

        assert config.SIMULATION_NAMES["generic_sims"] == "Our Simulations"
        assert config.SIMULATION_NAMES["lan_sims"] == "Lan et al. (2023)"
        assert config.SIMULATION_NAMES["tsui_sims"] == "Tsui et al. (2018)"

    def test_ablation_names(self) -> None:
        """Test ablation name mappings."""
        config = DataConfig()

        assert config.ABLATION_NAMES["base_subset"] == "Ablation Base Data"
        assert config.ABLATION_NAMES["superset"] == "Ablation Hemoglobin Superset"

    def test_real_names(self) -> None:
        """Test real dataset name mappings."""
        config = DataConfig()

        assert config.REAL_NAMES["pig_semantic"] == "Pig Semantic"
        assert config.REAL_NAMES["human"] == "Human"
        assert config.REAL_NAMES["pig_masks"] == "Pig Masks"

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = DataConfig()

        assert config.DEFAULT_SUBSAMPLE_SIZE_TISSUE_MODEL == 70000
        assert config.DEFAULT_RANDOM_SEED == 42
        assert config.DEFAULT_VAL_PERCENT == 0.1
        assert config.DEFAULT_TEST_PERCENT == 0.2
        assert config.DEFAULT_N_WAVELENGTHS == 100
        assert config.DEFAULT_N_LAYERS == 3
        assert config.DEFAULT_SUBSAMPLE_SIZE_SURROGATE_MODEL == 100000
        assert config.JACQUES_DEFAULT_REFRACTIVE_INDEX == 1.44

    def test_data_split_names(self) -> None:
        """Test data split names."""
        config = DataConfig()

        assert "train" in config.DATA_SPLIT_NAMES
        assert "val" in config.DATA_SPLIT_NAMES
        assert "test" in config.DATA_SPLIT_NAMES
        assert len(config.DATA_SPLIT_NAMES) == 3


class TestDataLoadersIntegration:
    """Integration tests for data loaders."""

    @pytest.mark.parametrize(  # type: ignore[misc]
        "dataloader", [HumanDataLoader, CombinedHumanDataLoader]
    )
    def test_real_data_loader_init(self, dataloader: type[DataLoader]) -> None:
        """Test real data loader initialization."""
        dataloader_instance = dataloader()

        assert dataloader_instance.dataset_name
        assert dataloader_instance.fold == "fold_0"

    @pytest.mark.parametrize(  # type: ignore[misc]
        "dataloader, data_type",
        [(HumanDataLoader, "human"), (CombinedHumanDataLoader, "combined")],
    )
    def test_real_data_loader_training_data(
        self, dataloader: type[DataLoader], data_type: str
    ) -> None:
        """Test real data loader training and validation dataloader return."""
        dataloader_instance = dataloader()
        train, val = dataloader_instance.load_training_data()

        assert isinstance(train, DataLoader)
        assert isinstance(val, DataLoader)
        if data_type == "human":
            assert len(train.dataset) == 411
            assert len(val.dataset) == 96
        elif data_type == "combined":
            assert len(train.dataset) == 2866
            assert len(val.dataset) == 675
        else:
            raise ValueError("Unsupported data_type")

    @pytest.mark.parametrize(  # type: ignore[misc]
        "dataloader, data_type",
        [(HumanDataLoader, "human"), (CombinedHumanDataLoader, "combined")],
    )
    def test_real_data_loader_test_data(
        self, dataloader: type[DataLoader], data_type: str
    ) -> None:
        """Test real data loader test dataloader return."""
        dataloader_instance = dataloader()
        test = dataloader_instance.load_test_data()

        assert isinstance(test, DataLoader)
        if data_type == "human":
            assert len(test.dataset) == 287
        elif data_type == "combined":
            assert len(test.dataset) == 1941
        else:
            raise ValueError("Unsupported data_type")

    @pytest.mark.parametrize(  # type: ignore[misc]
        "specular", [True, False]
    )
    def test_simulation_data_loader_init(self, specular: bool) -> None:
        """Test simulation data loader initialization."""
        dataloader_instance = SimulationDataLoaderManager(specular=specular)

        assert dataloader_instance.specular == specular
        assert dataloader_instance.loaders.keys() == {
            "generic",
            "lan",
            "tsui",
            "manoj",
            "jacques",
        }
