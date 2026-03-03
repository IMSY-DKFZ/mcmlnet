from unittest.mock import Mock

import numpy as np
import pytest
import torch

from mcmlnet.training.data_loading.datasets import (
    DatasetMCPred,
    DatasetMCPredDirect,
    mix_spectra_randomly,
)


class TestDatasetMCPredDirect:
    """Test cases for DatasetMCPredDirect class."""

    def setup_method(self) -> None:
        """Setup method to initialize common variables."""
        self.data = torch.randn(100, 3, 6)
        self.mock_preprocessor = Mock()
        self.mock_preprocessor.return_value = torch.randn(1, 3, 6)

    def test_init_valid(self) -> None:
        """Test valid initialization."""
        dataset = DatasetMCPredDirect(
            data=self.data.clone(),
            n_wavelengths=3,
            n_params=5,
            thick_deepest_layer=True,
            mix_data=True,
            preprocessor=None,
        )

        assert dataset.data.shape == self.data.shape
        assert dataset.n_wavelengths == 3
        assert dataset.n_params == 5
        assert dataset.thick_layer is True
        assert len(dataset) == 100

    @pytest.mark.parametrize("data", [[1, 2, 3], "invalid", np.random.rand(100, 3, 6)])  # type: ignore[misc]
    def test_init_invalid_data_type(self, data: list[int] | str | np.ndarray) -> None:
        """Test error with invalid data type."""
        with pytest.raises(TypeError, match=r"data must be torch.Tensor"):
            DatasetMCPredDirect(
                data=data,
                n_wavelengths=3,
                n_params=5,
                thick_deepest_layer=True,
                mix_data=True,
                preprocessor=None,
            )

    def test_init_invalid_n_wavelengths(self) -> None:
        """Test error with invalid n_wavelengths."""
        with pytest.raises(ValueError, match="n_wavelengths must be positive integer"):
            DatasetMCPredDirect(
                data=self.data.clone(),
                n_wavelengths=0,
                n_params=5,
                thick_deepest_layer=True,
                mix_data=True,
                preprocessor=None,
            )

        with pytest.raises(ValueError, match="n_wavelengths must be positive integer"):
            DatasetMCPredDirect(
                data=self.data.clone(),
                n_wavelengths=3.0,  # type: ignore[arg-type]
                n_params=5,
                thick_deepest_layer=True,
                mix_data=True,
                preprocessor=None,
            )

    def test_init_invalid_n_params(self) -> None:
        """Test error with invalid n_params."""
        with pytest.raises(ValueError, match="n_params must be positive integer"):
            DatasetMCPredDirect(
                data=self.data.clone(),
                n_wavelengths=3,
                n_params=0,
                thick_deepest_layer=True,
                mix_data=True,
                preprocessor=None,
            )

        with pytest.raises(ValueError, match="n_params must be positive integer"):
            DatasetMCPredDirect(
                data=self.data.clone(),
                n_wavelengths=3,
                n_params=5.0,  # type: ignore[arg-type]
                thick_deepest_layer=True,
                mix_data=True,
                preprocessor=None,
            )

    def test_init_invalid_thick_deepest_layer_type(self) -> None:
        """Test error with invalid thick_deepest_layer type."""
        with pytest.raises(TypeError, match="thick_deepest_layer must be bool"):
            DatasetMCPredDirect(
                data=self.data.clone(),
                n_wavelengths=3,
                n_params=5,
                thick_deepest_layer="True",  # type: ignore[arg-type]
                mix_data=True,
                preprocessor=None,
            )

    def test_init_invalid_mix_data_type(self) -> None:
        """Test error with invalid mix_data type."""
        with pytest.raises(TypeError, match="mix_data must be bool"):
            DatasetMCPredDirect(
                data=self.data.clone(),
                n_wavelengths=3,
                n_params=5,
                thick_deepest_layer=True,
                mix_data="True",  # type: ignore[arg-type]
                preprocessor=None,
            )

    def test_init_shape_mismatch(self) -> None:
        """Test error with shape mismatch."""
        with pytest.raises(ValueError, match="Individual data required to be of shape"):
            DatasetMCPredDirect(
                data=self.data.clone(),
                n_wavelengths=3,
                n_params=4,  # Expects 6 parameters (5 + 1)
                thick_deepest_layer=True,
                mix_data=True,
                preprocessor=None,
            )

    def test_init_dummy_shape_init(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test warning when 2D tensor is used for dataset init."""
        caplog.clear()
        caplog.set_level("WARNING")
        DatasetMCPredDirect(
            data=self.data.clone().reshape(100, -1),
            n_wavelengths=3,
            n_params=5,
            thick_deepest_layer=True,
            mix_data=True,
            preprocessor=None,
        )
        assert "Continuing with dummy 2D tensor for dataset init" in caplog.text

    def test_getitem_single_index(self) -> None:
        """Test __getitem__ with single index."""
        preprocessor = Mock()
        preprocessor.return_value = torch.randn(1, 3, 6)

        dataset = DatasetMCPredDirect(
            data=self.data.clone(),
            n_wavelengths=3,
            n_params=5,
            thick_deepest_layer=True,
            mix_data=True,
            preprocessor=preprocessor,
        )

        params, reflectance = dataset[0]

        assert isinstance(params, torch.Tensor)
        assert isinstance(reflectance, torch.Tensor)
        assert params.shape == (3, 5)  # 3 wavelengths, 5 parameters
        assert reflectance.shape == (3,)  # 3 wavelengths, 1 reflectance (squeezed)

    def test_getitem_list_index(self) -> None:
        """Test __getitem__ with list index."""
        preprocessor = Mock()
        preprocessor.return_value = torch.randn(2, 3, 6)

        dataset = DatasetMCPredDirect(
            data=self.data.clone(),
            n_wavelengths=3,
            n_params=5,
            thick_deepest_layer=True,
            mix_data=True,
            preprocessor=preprocessor,
        )

        params, reflectance = dataset[[0, 1]]

        assert isinstance(params, torch.Tensor)
        assert isinstance(reflectance, torch.Tensor)
        assert params.shape == (6, 5)
        assert reflectance.shape == (2, 3)

    def test_getitem_tensor_index(self) -> None:
        """Test __getitem__ with tensor index."""
        preprocessor = Mock()
        preprocessor.return_value = torch.randn(2, 3, 6)

        dataset = DatasetMCPredDirect(
            data=self.data.clone(),
            n_wavelengths=3,
            n_params=5,
            thick_deepest_layer=True,
            mix_data=True,
            preprocessor=preprocessor,
        )

        params, reflectance = dataset[torch.tensor([0, 1])]

        assert isinstance(params, torch.Tensor)
        assert isinstance(reflectance, torch.Tensor)
        assert params.shape == (6, 5)
        assert reflectance.shape == (2, 3)

    def test_getitem_no_preprocessor(self) -> None:
        """Test error when preprocessor is not provided."""
        dataset = DatasetMCPredDirect(
            data=self.data.clone(),
            n_wavelengths=3,
            n_params=5,
            thick_deepest_layer=True,
            mix_data=True,
            preprocessor=None,
        )

        with pytest.raises(
            ValueError, match="Preprocessor is required but not provided"
        ):
            dataset[0]


class TestDatasetMCPred:
    """Test cases for DatasetMCPred class."""

    def setup_method(self) -> None:
        """Setup method to initialize common variables."""
        self.data = torch.randn(100, 3, 6)

    def test_getitem(self) -> None:
        """Test __getitem__ method."""
        dataset = DatasetMCPred(
            data=self.data.clone(),
            n_wavelengths=3,
            n_params=5,
            thick_deepest_layer=True,
            mix_data=True,
            preprocessor=None,
        )

        result = dataset[0]

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 6)

    def test_getitem_list_index(self) -> None:
        """Test __getitem__ with list index."""
        dataset = DatasetMCPred(
            data=self.data.clone(),
            n_wavelengths=3,
            n_params=5,
            thick_deepest_layer=True,
            mix_data=True,
            preprocessor=None,
        )

        result = dataset[[0, 1]]

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 3, 6)

    def test_getitem_tensor_index(self) -> None:
        """Test __getitem__ with tensor index."""
        dataset = DatasetMCPred(
            data=self.data.clone(),
            n_wavelengths=3,
            n_params=5,
            thick_deepest_layer=True,
            mix_data=True,
            preprocessor=None,
        )

        result = dataset[torch.tensor([0, 1])]

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 3, 6)


class TestMixSpectraRandomly:
    """Test cases for mix_spectra_randomly function."""

    def setup_method(self) -> None:
        """Setup method to initialize common variables."""
        self.spectra = torch.randn(10, 3, 5)

    def test_mix_spectra_randomly_true(self) -> None:
        """Test mixing spectra when mix_data=True."""
        spectra = self.spectra.clone()

        mix_spectra_randomly(spectra, mix_data=True)

        assert not torch.allclose(spectra, self.spectra)

    def test_mix_spectra_randomly_false(self) -> None:
        """Test not mixing spectra when mix_data=False."""
        spectra = self.spectra.clone()

        mix_spectra_randomly(spectra, mix_data=False)

        assert torch.allclose(spectra, self.spectra)

    @pytest.mark.parametrize(
        "spectra", [[1, 2, 3], "invalid", np.random.rand(10, 3, 5)]
    )  # type: ignore[misc]
    def test_mix_spectra_randomly_invalid_type(
        self, spectra: list[int] | str | np.ndarray
    ) -> None:
        """Test error with invalid tensor type."""
        with pytest.raises(TypeError, match=r"spectra must be torch.Tensor"):
            mix_spectra_randomly(spectra, mix_data=True)

    @pytest.mark.parametrize("spectra", [torch.randn(10, 5), torch.randn(10, 3, 3, 5)])  # type: ignore[misc]
    def test_mix_spectra_randomly_invalid_dim(self, spectra: torch.Tensor) -> None:
        """Test error with invalid tensor dimensions."""
        with pytest.raises(ValueError, match="Input tensor must be 3-dimensional"):
            mix_spectra_randomly(spectra, mix_data=True)

    def test_mix_spectra_randomly_zero_dimensions(self) -> None:
        """Test error with zero dimensions."""
        for dim in range(self.spectra.ndim):
            shape = list(self.spectra.shape)
            shape[dim] = 0
            spectra = torch.randn(*shape)

            with pytest.raises(
                ValueError, match="Input tensor must have non-zero dimensions"
            ):
                mix_spectra_randomly(spectra, mix_data=True)
