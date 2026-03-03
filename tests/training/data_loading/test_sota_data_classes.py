"""Tests for mcmlnet.training.data_loading.sota_data_classes module."""

from itertools import product

import pytest
import torch

from mcmlnet.training.data_loading.sota_data_classes import (
    SOTADataset,
    SOTAPreprocessor,
)


class TestSOTAPreprocessor:
    """Test cases for SOTAPreprocessor class."""

    @pytest.mark.parametrize(  # type: ignore[misc]
        "dataset_name,n_wavelengths,expected_shape",
        [
            ("lan_lhs", 1, (5000, 1, 5)),
            ("lan", 1, (5000, 1, 5)),
            ("tsui", 1, (30000, 1, 20)),
            ("manoj", 351, (35000, 351, 10)),
        ],
    )
    def test_init_default_parameters(
        self,
        dataset_name: str,
        n_wavelengths: int,
        expected_shape: tuple[int, int, int],
    ) -> None:
        """Test initialization with default parameters."""
        preprocessor = SOTAPreprocessor(
            dataset_name=dataset_name, n_wavelengths=n_wavelengths
        )

        # Check attributes
        assert preprocessor.dataset_name == dataset_name
        assert preprocessor.n_wavelengths == n_wavelengths
        assert preprocessor.log_intensity is False
        assert preprocessor.n_pca_comp == 0
        assert preprocessor.kfolds is None
        assert preprocessor.fold is None
        assert preprocessor.norm_1 is None
        assert preprocessor.norm_2 is None

        # Check data shapes
        assert preprocessor.params.shape == expected_shape
        assert preprocessor.labels.shape == (expected_shape[0], expected_shape[1], 1)

    def test_init_invalid_dataset(self) -> None:
        """Test error with invalid dataset name."""
        with pytest.raises(ValueError, match="Dataset 'invalid' not implemented"):
            SOTAPreprocessor(dataset_name="invalid", n_wavelengths=1)

    def test_init_invalid_n_wavelengths_lan(self) -> None:
        """Test error with invalid n_wavelengths for Lan et al. dataset."""
        with pytest.raises(
            ValueError, match="LAN dataset only supports n_wavelengths=1"
        ):
            SOTAPreprocessor(dataset_name="lan", n_wavelengths=2)

        with pytest.raises(
            ValueError, match="LAN LHS dataset only supports n_wavelengths=1"
        ):
            SOTAPreprocessor(dataset_name="lan_lhs", n_wavelengths=2)

    def test_init_invalid_n_wavelengths_tsui(self) -> None:
        """Test error with invalid n_wavelengths for Tsui et al. dataset."""
        with pytest.raises(
            ValueError, match="TSUI dataset only supports n_wavelengths=1"
        ):
            SOTAPreprocessor(dataset_name="tsui", n_wavelengths=2)

    def test_init_invalid_n_wavelengths_manoj(self) -> None:
        """Test error with invalid n_wavelengths for Manojlovic et al. dataset."""
        with pytest.raises(
            ValueError, match="MANOJLOVIC dataset only supports n_wavelengths=351"
        ):
            SOTAPreprocessor(dataset_name="manoj", n_wavelengths=100)

    @pytest.mark.parametrize(  # type: ignore[misc]
        "dataset_name", ["lan_lhs", "lan"]
    )
    def test_preprocess_lan_data(self, dataset_name: str) -> None:
        """Test preprocessing LAN data."""
        # Create test data
        preprocessor = SOTAPreprocessor(dataset_name=dataset_name, n_wavelengths=1)
        preprocessor.params = torch.arange(1.0, 6.0).reshape(1, 1, 5)

        # Apply preprocessing
        preprocessor._preprocess_lan_data()

        # Check that log10 was applied to mu_a, mu_s, and d
        assert torch.allclose(
            preprocessor.params[0, 0, :2], torch.log10(torch.tensor([1.0, 2.0]))
        )
        assert torch.allclose(
            preprocessor.params[0, 0, -1], torch.log10(torch.tensor([5.0]))
        )

    def test_preprocess_tsui_data(self) -> None:
        """Test preprocessing TSUI data."""
        # Create test data with multiple layers
        preprocessor = SOTAPreprocessor(dataset_name="tsui", n_wavelengths=1)
        preprocessor.params = torch.arange(1.0, 21.0).reshape(1, 1, 20)
        print(preprocessor.params)
        print(torch.log10(preprocessor.params))

        # Apply preprocessing
        preprocessor._preprocess_tsui_data()
        print(preprocessor.params)

        # Check that log10 was applied to mu_a, mu_s, and d for each layer
        assert torch.allclose(
            preprocessor.params[0, 0, [0, 1]], torch.log10(torch.tensor([1.0, 2.0]))
        )
        assert torch.allclose(
            preprocessor.params[0, 0, [4, 5, 6]],
            torch.log10(torch.tensor([5.0, 6.0, 7.0])),
        )
        assert torch.allclose(
            preprocessor.params[0, 0, [9, 10, 11]],
            torch.log10(torch.tensor([10.0, 11.0, 12.0])),
        )
        assert torch.allclose(
            preprocessor.params[0, 0, [14, 15, 16]],
            torch.log10(torch.tensor([15.0, 16.0, 17.0])),
        )
        assert torch.allclose(
            preprocessor.params[0, 0, [19]], torch.log10(torch.tensor([20.0]))
        )

    def test_preprocess_manoj_data(self) -> None:
        """Test preprocessing Manoj data."""
        # Create test data
        preprocessor = SOTAPreprocessor(dataset_name="manoj", n_wavelengths=351)
        preprocessor.params = torch.arange(1.0, 11.0).reshape(1, 1, 10)

        # Apply preprocessing
        preprocessor._preprocess_manoj_data()

        # Check that log10 was applied to mu_a and mu_s for both layers
        assert torch.allclose(
            preprocessor.params[:, :, [0, 1, 5, 6]],
            torch.log10(torch.tensor([1.0, 2.0, 6.0, 7.0])),
        )

    def test_fit(self) -> None:
        """Test fit method."""
        # Create test data
        preprocessor = SOTAPreprocessor(dataset_name="lan_lhs", n_wavelengths=1)
        preprocessor.params = torch.randn(100, 1, 5)
        preprocessor.labels = torch.randn(100, 1, 1)
        result = preprocessor.fit()

        # Should return concatenated data
        assert result.shape == (100, 1, 6)
        assert torch.allclose(result[..., :-1], preprocessor.params)
        assert torch.allclose(result[..., -1:], preprocessor.labels)

    def test_fit_2d(self) -> None:
        """Test fit method."""
        # Create test data
        preprocessor = SOTAPreprocessor(dataset_name="lan_lhs", n_wavelengths=1)
        preprocessor.params = torch.randn(100, 5)
        preprocessor.labels = torch.randn(100, 1)
        result = preprocessor.fit()

        # Should return concatenated data
        assert result.shape == (100, 6)
        assert torch.allclose(result[..., :-1], preprocessor.params)
        assert torch.allclose(result[..., -1:], preprocessor.labels)

    def test_call(self) -> None:
        """Test __call__ method."""
        preprocessor = SOTAPreprocessor(dataset_name="lan_lhs", n_wavelengths=1)
        preprocessor.norm_1 = torch.randn(5)
        preprocessor.norm_2 = torch.randn(5)

        # Create test data
        data = torch.randn(10, 1, 6)

        result = preprocessor(data)

        # Should apply normalization and clamping
        assert result.shape == (10, 1, 6)
        assert torch.all(result[..., -1] >= 0) and torch.all(result[..., -1] <= 1)

    def test_call_invalid_shape(self) -> None:
        """Test error with invalid data shape."""
        preprocessor = SOTAPreprocessor(dataset_name="lan_lhs", n_wavelengths=1)

        # Create invalid data shape
        data = torch.randn(10, 6)

        with pytest.raises(ValueError, match="Data must be 3D tensor"):
            preprocessor(data)

    def test_consistent_data_split_ids(self) -> None:
        """Test default consistent data split IDs."""
        preprocessor = SOTAPreprocessor(dataset_name="lan_lhs", n_wavelengths=1)

        dummy_data = torch.randn(100, 1, 6)

        # Test train split
        train_ids = preprocessor.consistent_data_split_ids(dummy_data, "train")
        assert len(train_ids) == 70  # 70% of 100 samples

        # Test validation split
        val_ids = preprocessor.consistent_data_split_ids(dummy_data, "val")
        assert len(val_ids) == 10  # 10% of 100 samples

        # Test test split
        test_ids = preprocessor.consistent_data_split_ids(dummy_data, "test")
        assert len(test_ids) == 20  # 20% of 100 samples

        # Test invalid mode
        with pytest.raises(ValueError, match="Mode must be"):
            preprocessor.consistent_data_split_ids(dummy_data, "invalid")


class TestSOTADataset:
    """Test cases for SOTADataset class."""

    @staticmethod
    def dummy_dataset(data: torch.Tensor) -> SOTADataset:
        return SOTADataset(
            data=data,
            n_wavelengths=1,
            n_params=8,
            thick_deepest_layer=True,
            preprocessor=None,
        )

    @pytest.mark.parametrize(  # type: ignore[misc]
        "n_wavelengths,n_params,thick_deepest_layer",
        list(product([1, 351], [5, 10, 20], [True, False])),
    )
    def test_init_valid(
        self, n_wavelengths: int, n_params: int, thick_deepest_layer: bool
    ) -> None:
        """Test valid initialization."""
        data = torch.randn(100, n_wavelengths, n_params + 1)

        dataset = SOTADataset(
            data=data,
            n_wavelengths=n_wavelengths,
            n_params=n_params,
            thick_deepest_layer=thick_deepest_layer,
            preprocessor=None,
        )

        assert dataset.data.shape == (100, n_wavelengths, n_params + 1)
        assert dataset.n_wvl == n_wavelengths
        assert dataset.n_params == n_params
        assert dataset.thick_deepest_layer is thick_deepest_layer
        assert len(dataset) == 100

    def test_init_invalid_shape(self) -> None:
        """Test error with invalid data shape."""
        data = torch.randn(100, 9)  # 2D instead of 3D

        with pytest.raises(ValueError, match="Data must be 3D tensor"):
            self.dummy_dataset(data)

    def test_getitem(self) -> None:
        """Test __getitem__ method."""
        dataset = self.dummy_dataset(torch.randn(100, 1, 9))

        item = dataset[0]
        assert item.shape == (1, 9)
        items = dataset[[0, 1, 2]]
        assert items.shape == (3, 1, 9)
        tensor_idx = torch.tensor([0, 1])
        items = dataset[tensor_idx]
        assert items.shape == (2, 1, 9)

    def test_getitem_invalid_index(self) -> None:
        """Test error with invalid out of bounds indices."""
        dataset = self.dummy_dataset(torch.randn(100, 1, 9))

        with pytest.raises(IndexError):
            dataset[100]
        with pytest.raises(IndexError):
            dataset[torch.tensor([100])]
