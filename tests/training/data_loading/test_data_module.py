"""Tests for mcmlnet.training.data_loading.data_module and datasets modules."""

from unittest.mock import Mock, patch

import pytest
import torch

from mcmlnet.training.data_loading.data_module import DataModule


class TestDataModule:
    """Test cases for DataModule class."""

    def setup_method(self) -> None:
        """Setup for tests."""
        # Mock configuration
        self.cfg = Mock()
        self.cfg.n_workers = 4
        self.cfg.batch_size = 32
        self.cfg.max_augm_epochs = 10
        self.cfg.dataset = Mock()
        self.cfg.dataset.thick_deepest_layer = True
        self.cfg.dataset.n_params = 5
        self.cfg.preprocessing = Mock()
        self.cfg.preprocessing.kfolds = None
        self.cfg.preprocessing.fold = 0
        self.cfg.additional_scaling = {}
        self.cfg.get = Mock(return_value=None)

        # Mock preprocessor
        self.mock_preprocessor = Mock()
        self.mock_preprocessor.fit.return_value = torch.randn(100, 1, 10)
        self.mock_preprocessor.consistent_data_split_ids.side_effect = [
            list(range(70)),  # train
            list(range(70, 80)),  # val
            list(range(80, 100)),  # test
        ]

    def test_init_valid(self) -> None:
        """Test valid initialization."""
        with patch(
            "mcmlnet.training.data_loading.data_module.instantiate",
            return_value=self.mock_preprocessor,
        ):
            data_module = DataModule(self.cfg)

        assert data_module.n_workers == 4
        assert data_module.batch_size == 32
        assert data_module.max_augm_epochs == 10
        assert data_module.cfg == self.cfg

    def test_init_invalid_config(self) -> None:
        """Test error with invalid configuration."""
        cfg = Mock()
        cfg.n_workers = -1

        with pytest.raises(ValueError, match="n_workers must be non-negative integer"):
            DataModule(cfg)

        cfg = Mock()
        cfg.n_workers = 4
        cfg.batch_size = 0

        with pytest.raises(ValueError, match="batch_size must be positive integer"):
            DataModule(cfg)

        cfg = Mock()
        cfg.n_workers = 4
        cfg.batch_size = 32
        cfg.max_augm_epochs = -5

        with pytest.raises(
            ValueError, match="max_augm_epochs must be non-negative integer"
        ):
            DataModule(cfg)

    def test_create_datasets(self) -> None:
        """Test dataset creation."""
        # Mock dataset class
        mock_dataset = Mock()

        with patch(
            "mcmlnet.training.data_loading.data_module.instantiate",
            side_effect=[
                self.mock_preprocessor,
                mock_dataset,
                mock_dataset,
                mock_dataset,
            ],
        ):
            data_module = DataModule(self.cfg)

        assert hasattr(data_module, "train_")
        assert hasattr(data_module, "val_")
        assert hasattr(data_module, "test_")

    @pytest.mark.parametrize("use_only_3M_data", [True, False])  # type: ignore[misc]
    def test_add_scaling_datasets(self, use_only_3M_data: bool) -> None:
        """Test adding scaling datasets."""
        cfg = self.cfg
        cfg.additional_scaling = {
            "dataset_loader": {"_target_": "test.loader"},
            "data": "test_data",
            "n_wavelengths": 100,
            "use_only_3M_data": use_only_3M_data,
        }

        # Mock dataset class
        mock_dataset = Mock()

        # Mock scaling dataset loader
        mock_scaling_loader = Mock()
        mock_scaling_loader.load_data.return_value = torch.randn(50, 1, 10)

        with patch(
            "mcmlnet.training.data_loading.data_module.instantiate",
            side_effect=[
                self.mock_preprocessor,
                mock_dataset,
                mock_dataset,
                mock_dataset,  # main datasets
                mock_scaling_loader,
                mock_dataset,
                mock_dataset,
                mock_dataset,  # scaling datasets
            ],
        ):
            data_module = DataModule(cfg)

        assert hasattr(data_module, "train_")
        assert hasattr(data_module, "val_")
        assert hasattr(data_module, "test_")

    def test_create_dataloader(self) -> None:
        """Test dataloader creation."""
        # Mock dataset class
        mock_dataset = Mock()
        mock_dataset.__len__ = lambda self: 70

        with patch(
            "mcmlnet.training.data_loading.data_module.instantiate",
            side_effect=[
                self.mock_preprocessor,
                mock_dataset,
                mock_dataset,
                mock_dataset,
            ],
        ):
            data_module = DataModule(self.cfg)

        # Test creating dataloaders
        train_dl = data_module.train_dataloader()
        val_dl = data_module.val_dataloader()
        test_dl = data_module.test_dataloader()

        assert train_dl is not None
        assert val_dl is not None
        assert test_dl is not None
        assert isinstance(train_dl, torch.utils.data.DataLoader)
        assert isinstance(val_dl, torch.utils.data.DataLoader)
        assert isinstance(test_dl, torch.utils.data.DataLoader)
