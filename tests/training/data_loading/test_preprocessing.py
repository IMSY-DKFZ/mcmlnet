"""Tests for mcmlnet.training.data_loading.preprocessing module."""

from itertools import product
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from mcmlnet.training.data_loading.preprocessing import (
    PreProcessor,
    collate_variable_tensors,
    collate_variable_tensors_old,
    process_2d_3d_data,
    set_deepest_layer_to_zero,
    subsample_wavelengths,
)


class TestSubsampleWavelengths:
    """Test cases for subsample_wavelengths function."""

    @pytest.mark.parametrize(  # type: ignore[misc]
        "data", [torch.randn(10, 24 + 351), np.random.randn(10, 24 + 351)]
    )
    def test_subsample_wavelengths_valid(self, data: torch.Tensor | np.ndarray) -> None:
        """Test valid wavelength subsampling."""
        result = subsample_wavelengths(data, n_wvl=100)

        assert result.shape == (10, 24 + 100)
        if isinstance(data, torch.Tensor):
            assert torch.allclose(result[:, :24], data[:, :24])
        elif isinstance(data, np.ndarray):
            assert np.allclose(result[:, :24], data[:, :24])
        else:
            raise TypeError("Data must be torch.Tensor or np.ndarray")

    def test_subsample_wavelengths_invalid_n_wvl(self) -> None:
        """Test error with invalid n_wvl."""
        data = torch.randn(10, 24 + 351)

        with pytest.raises(ValueError, match="n_wvl must be a positive integer"):
            subsample_wavelengths(data, n_wvl=0)

        with pytest.raises(ValueError, match="n_wvl must be a positive integer"):
            subsample_wavelengths(data, n_wvl=400)

        with pytest.raises(ValueError, match="n_wvl must be a positive integer"):
            subsample_wavelengths(data, n_wvl=4.0)  # type: ignore[arg-type]

    def test_subsample_wavelengths_invalid_type(self) -> None:
        """Test error with invalid data type."""
        with pytest.raises(
            TypeError, match=r"simulations must be torch.Tensor or np.ndarray"
        ):
            subsample_wavelengths([1, 2, 3], n_wvl=100)


class TestPreProcessor:
    """Test cases for PreProcessor class."""

    def setup_method(self) -> None:
        """Setup method to initialize common variables."""
        self.data_2d = torch.randn(100, 10)

    def test_init_default_parameters(self) -> None:
        """Test initialization with default parameters."""
        preprocessor = PreProcessor()

        assert preprocessor.dataset_name == "reflectance"
        assert preprocessor.n_wavelengths == 100
        assert preprocessor.val_percent == 0.1
        assert preprocessor.test_percent == 0.2
        assert preprocessor.is_or_make_physical is True
        assert preprocessor.log_parameters is False
        assert preprocessor.n_layers == 3
        assert preprocessor.refl_range == "None"
        assert preprocessor.param_range is None
        assert preprocessor.kfolds is None
        assert preprocessor.fold is None
        assert preprocessor.log_intensity is False
        assert preprocessor.n_pca_comp is None
        assert preprocessor.batch_size is None
        assert preprocessor.wavelengths is None
        assert preprocessor.norm_1 is None
        assert preprocessor.norm_2 is None

    def test_init_custom_parameters(self) -> None:
        """Test initialization with custom parameters."""
        preprocessor = PreProcessor(
            dataset_name="test_dataset",
            n_wavelengths=200,
            val_percent=0.15,
            test_percent=0.25,
            is_or_make_physical=False,
            log=True,
            n_layers=5,
            refl_range="asym",
            param_range="z-score",
            kfolds=5,
            fold=2,
            log_intensity=True,
            n_pca_comp=10,
            batch_size=64,
            wavelengths=np.linspace(400, 800, 100),
            norm_1=torch.randn(10),
            norm_2=torch.randn(10),
        )

        assert preprocessor.dataset_name == "test_dataset"
        assert preprocessor.n_wavelengths == 200
        assert preprocessor.val_percent == 0.15
        assert preprocessor.test_percent == 0.25
        assert preprocessor.is_or_make_physical is False
        assert preprocessor.log_parameters is True
        assert preprocessor.n_layers == 5
        assert preprocessor.refl_range == "asym"
        assert preprocessor.param_range == "z-score"
        assert preprocessor.kfolds == 5
        assert preprocessor.fold == 2
        assert preprocessor.log_intensity is True
        assert preprocessor.n_pca_comp == 10
        assert preprocessor.batch_size == 64
        assert preprocessor.wavelengths is not None
        assert preprocessor.norm_1 is not None
        assert preprocessor.norm_2 is not None

    def test_init_invalid_parameters(self) -> None:
        """Test error with invalid parameters."""
        with pytest.raises(ValueError, match="n_wavelengths must be positive integer"):
            PreProcessor(n_wavelengths=0)

        with pytest.raises(ValueError, match="val_percent must be between 0 and 1"):
            PreProcessor(val_percent=1.5)
        with pytest.raises(ValueError, match="val_percent must be between 0 and 1"):
            PreProcessor(val_percent=-0.1)

        with pytest.raises(ValueError, match="test_percent must be between 0 and 1"):
            PreProcessor(test_percent=1.5)
        with pytest.raises(ValueError, match="test_percent must be between 0 and 1"):
            PreProcessor(test_percent=-0.1)

        with pytest.raises(
            ValueError, match="val_percent \\+ test_percent must be less than 1"
        ):
            PreProcessor(val_percent=0.6, test_percent=0.5)

        with pytest.raises(ValueError, match="n_layers must be positive integer"):
            PreProcessor(n_layers=0)
        with pytest.raises(ValueError, match="n_layers must be positive integer"):
            PreProcessor(n_layers=3.0)  # type: ignore[arg-type]

    def test_consistent_data_split_ids_no_kfolds(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test data splitting without k-folds."""
        preprocessor = PreProcessor(val_percent=0.1, test_percent=0.2)

        caplog.clear()
        caplog.set_level("WARNING")
        train_ids = preprocessor.consistent_data_split_ids(
            self.data_2d.clone(), "train"
        )
        assert len(train_ids) == 70
        assert "No k-folds specified! Using single training split." in caplog.text
        caplog.clear()
        caplog.set_level("WARNING")
        val_ids = preprocessor.consistent_data_split_ids(self.data_2d.clone(), "val")
        assert len(val_ids) == 10
        assert caplog.text == ""
        caplog.clear()
        caplog.set_level("WARNING")
        test_ids = preprocessor.consistent_data_split_ids(self.data_2d.clone(), "test")
        assert len(test_ids) == 20
        assert caplog.text == ""

    def test_consistent_data_split_ids_with_kfolds(self) -> None:
        """Test data splitting with k-folds."""
        preprocessor = PreProcessor(val_percent=0.1, test_percent=0.2)

        train_ids = preprocessor.consistent_data_split_ids(
            self.data_2d.clone(), "train", kfolds=3, fold=0
        )
        assert len(train_ids) == int(len(self.data_2d) * 0.7 * (2 / 3))

    def test_consistent_data_split_ids_invalid_mode(self) -> None:
        """Test error with invalid split mode."""
        preprocessor = PreProcessor()

        with pytest.raises(NotImplementedError, match="Mode invalid not implemented"):
            preprocessor.consistent_data_split_ids(self.data_2d.clone(), "invalid")

    def test_consistent_data_split_ids_kfold_error(self) -> None:
        """Test error with k-fold parameters."""
        preprocessor = PreProcessor()

        with pytest.raises(AssertionError, match="Fold must be specified"):
            preprocessor.consistent_data_split_ids(
                self.data_2d.clone(), "train", kfolds=3
            )
        with pytest.raises(
            AssertionError, match=r"Fold 3 not found in 3 splits \(0-indexed\)."
        ):
            preprocessor.consistent_data_split_ids(
                self.data_2d.clone(), "train", kfolds=3, fold=3
            )

    def test_apply_log10_to_mua_mus_d(self) -> None:
        """Test log10 transformation."""
        preprocessor = PreProcessor(n_layers=2)

        physical_parameters = torch.tensor(
            [
                [
                    [
                        1.0,
                        2.0,  # mu_a
                        3.0,
                        4.0,  # mu_s
                        0.8,
                        0.9,  # g
                        1.35,
                        1.35,  # n
                        0.001,
                        0.002,  # d
                    ]
                ]
                * 3
            ]
        )

        result = preprocessor._apply_log10_to_mua_mus_d(physical_parameters.clone())

        # Check that log10 was applied to mu_a, mu_s, and d
        assert torch.allclose(result[0, 0, :2], torch.log10(torch.tensor([1.0, 2.0])))
        assert torch.allclose(result[0, 0, 2:4], torch.log10(torch.tensor([3.0, 4.0])))
        assert torch.allclose(
            result[0, 0, 8:10], torch.log10(torch.tensor([0.001, 0.002]))
        )
        # Check that g and n were not changed
        assert torch.allclose(result[0, 0, 4:6], torch.tensor([0.8, 0.9]))
        assert torch.allclose(result[0, 0, 6:8], torch.tensor([1.35, 1.35]))

    @pytest.mark.parametrize(  # type: ignore[misc]
        "data", [[1, 2, 3], np.random.randn(10, 5)]
    )
    def test_apply_log10_to_mua_mus_d_invalid_type(
        self, data: list | np.ndarray
    ) -> None:
        """Test error with invalid tensor type."""
        preprocessor = PreProcessor(n_layers=2)

        with pytest.raises(TypeError, match=r"parameter_tensor must be torch.Tensor"):
            preprocessor._apply_log10_to_mua_mus_d(data)

    def test_apply_log10_to_mua_mus_d_invalid_shape(self) -> None:
        """Test error with invalid tensor dimensionality."""
        preprocessor = PreProcessor(n_layers=2)
        with pytest.raises(ValueError, match="parameter_tensor must be 3D"):
            preprocessor._apply_log10_to_mua_mus_d(torch.randn(10, 10))

    def test_apply_log10_to_mua_mus_d_invalid_parameter_shape(self) -> None:
        """Test error with invalid number of parameters."""
        preprocessor = PreProcessor(n_layers=2)
        with pytest.raises(
            ValueError, match="Number of parameters must be divisible by n_layers"
        ):
            preprocessor._apply_log10_to_mua_mus_d(torch.randn(1, 1, 9))

    @pytest.mark.parametrize(  # type: ignore[misc]
        "is_or_make_physical,log,n_layers,param_range,batch_size",
        list(
            product(
                [True, False],
                [True, False],
                [1, 2, 3],
                ["z-score", "None", None],
                [None, 10, 1000],
            )
        ),
    )
    def test_fit(
        self,
        is_or_make_physical: bool,
        log: bool,
        n_layers: int,
        param_range: str | None,
        batch_size: int | None,
    ) -> None:
        """Test fit method."""
        with (
            patch(
                "mcmlnet.training.data_loading.preprocessing.SimulationDataLoader"
            ) as mock_loader_class,
            patch.object(
                PreProcessor, "consistent_data_split_ids", return_value=list(range(70))
            ),
        ):
            # Mock data loader
            mock_loader = Mock()
            mock_loader.load_data.return_value = torch.randn(100, 8 * n_layers + 100)
            mock_loader_class.return_value = mock_loader

            preprocessor = PreProcessor(
                n_wavelengths=100,
                is_or_make_physical=is_or_make_physical,
                log=log,
                n_layers=n_layers,
                param_range=param_range,
                batch_size=batch_size,
            )
            result = preprocessor.fit()

            assert result.shape == (100, 8 * n_layers + 100)
            assert preprocessor.is_or_make_physical == is_or_make_physical
            assert preprocessor.log_parameters == log
            mock_loader.load_data.assert_called_once()

            if param_range in ["None", None]:
                assert preprocessor.param_range is param_range
                assert preprocessor.norm_1 is None
                assert preprocessor.norm_2 is None
            else:
                if is_or_make_physical:
                    n_params = n_layers * 5  # mu_a, mu_s, g, n, d
                else:
                    n_params = n_layers * 8  # mu_a, mu_s, g, n, d, reff, v, f
                assert preprocessor.norm_1.shape == (n_params,)  # type: ignore[union-attr]
                assert preprocessor.norm_2.shape == (n_params,)  # type: ignore[union-attr]

    @pytest.mark.parametrize(  # type: ignore[misc]
        "is_or_make_physical,log,n_layers,param_range,log_intensity",
        list(
            product(
                [True, False],
                [True, False],
                [1, 2, 3],
                ["z-score", "None", None],
                [True, False],
            )
        ),
    )
    def test_call(
        self,
        is_or_make_physical: bool,
        log: bool,
        n_layers: int,
        param_range: str | None,
        log_intensity: bool,
    ) -> None:
        """Test __call__ method."""
        preprocessor = PreProcessor(
            n_wavelengths=100,
            is_or_make_physical=is_or_make_physical,
            log=log,
            n_layers=n_layers,
            param_range=param_range,
            log_intensity=log_intensity,
        )

        if is_or_make_physical:
            n_params = n_layers * 5
        else:
            n_params = n_layers * 8

        if param_range in ["None", None]:
            preprocessor.norm_1 = None
            preprocessor.norm_2 = None
        else:
            preprocessor.norm_1 = torch.randn(1, 1, n_params)
            preprocessor.norm_2 = torch.randn(1, 1, n_params)

        result = preprocessor(torch.randn(50, 8 * n_layers + 100))

        if is_or_make_physical:
            assert result.shape == (50, 100, n_params + 1)
        else:
            assert result.shape == (50, n_params + 100)


class TestUtilityFunctions:
    """Test cases for utility functions."""

    def setup_method(self) -> None:
        """Setup method to initialize common variables."""
        self.data_2d = torch.randn(10, 5)

    @pytest.mark.parametrize(  # type: ignore[misc]
        "data,index",
        list(
            product(
                [
                    torch.randn(10, 5),
                    torch.randn(10, 3, 5),
                    np.random.randn(10, 5),
                    np.random.randn(10, 3, 5),
                ],
                [0, [0, 1], torch.tensor([0, 1])],
            )
        ),
    )
    def test_process_2d_3d_valid(
        self, data: torch.Tensor | np.ndarray, index: int | list[int] | torch.Tensor
    ) -> None:
        """Test process_2d_3d_data with 2D and 3D data."""
        preprocessor = Mock()
        preprocessor.return_value = torch.randn(
            1 if isinstance(index, int) else len(index), 3, 5
        )

        result = process_2d_3d_data(data, index, preprocessor)

        assert result.shape == (1 if isinstance(index, int) else len(index), 3, 5)

    def test_process_2d_3d_data_invalid_dim(self) -> None:
        """Test error with invalid data dimensions."""
        with pytest.raises(ValueError, match="Data tensor must be 2D or 3D"):
            process_2d_3d_data(torch.randn(10, 3, 4, 5), 0, Mock())

    def test_process_2d_3d_data_invalid_index(self) -> None:
        """Test error with invalid index."""
        with pytest.raises(ValueError, match="Index 100 out of bounds"):
            process_2d_3d_data(self.data_2d, 100, Mock())

    def test_process_2d_3d_data_invalid_index_list(self) -> None:
        """Test error with invalid index."""
        with pytest.raises(
            ValueError, match=r"Some indices in \[5, 100\] out of bounds"
        ):
            process_2d_3d_data(self.data_2d, [5, 100], Mock())

        with pytest.raises(
            TypeError, match="All elements in index list must be integer"
        ):
            process_2d_3d_data(self.data_2d, [5.0, 6], Mock())

    def test_process_2d_3d_data_invalid_type(self) -> None:
        """Test error with invalid data type."""
        with pytest.raises(
            TypeError, match=r"index must be int, list\[int\], or torch.Tensor"
        ):
            process_2d_3d_data(self.data_2d, "0", Mock())

    def test_set_deepest_layer_to_zero_true(self) -> None:
        """Test set_deepest_layer_to_zero with thick_layer=True."""
        original_value = self.data_2d.clone()

        result = set_deepest_layer_to_zero(self.data_2d.clone(), thick_layer=True)

        assert torch.all(result[:, -2] == 0)
        assert torch.allclose(result[:, :-2], original_value[:, :-2])
        assert torch.allclose(result[:, -1], original_value[:, -1])

    def test_set_deepest_layer_to_zero_false(self) -> None:
        """Test set_deepest_layer_to_zero with thick_layer=False."""
        original_data = self.data_2d.clone()

        result = set_deepest_layer_to_zero(self.data_2d.clone(), thick_layer=False)

        assert torch.allclose(result, original_data)

    @pytest.mark.parametrize(  # type: ignore[misc]
        "data,thick_layer",
        list(product([[1, 2, 3], np.random.randn(10, 5)], [True, False])),
    )
    def test_set_deepest_layer_to_zero_invalid_type(
        self, data: list[int] | np.ndarray, thick_layer: bool
    ) -> None:
        """Test error with invalid data type."""
        with pytest.raises(TypeError, match=r"data must be torch.Tensor"):
            set_deepest_layer_to_zero(data, thick_layer=thick_layer)

    @pytest.mark.parametrize(  # type: ignore[misc]
        "thick_layer",
        [[1, 2, 3], np.random.randn(10, 5)],
    )
    def test_set_deepest_layer_to_zero_invalid_thick_layer_type(
        self, thick_layer: list[int] | np.ndarray
    ) -> None:
        """Test error with invalid data type."""
        with pytest.raises(TypeError, match=r"thick_layer must be bool, got"):
            set_deepest_layer_to_zero(self.data_2d, thick_layer=thick_layer)  # type: ignore[arg-type]

    def test_collate_variable_tensors_old(self) -> None:
        """Test collate_variable_tensors_old function."""
        batch = [
            (torch.randn(5, 3), torch.randn(5, 2)),
            (torch.randn(3, 3), torch.randn(3, 2)),
            (torch.randn(7, 3), torch.randn(7, 2)),
        ]

        inputs, targets = collate_variable_tensors_old(batch)

        assert inputs.shape == (15, 3)
        assert targets.shape == (15, 2)

    @pytest.mark.parametrize(  # type: ignore[misc]
        "layer_thick",
        [True, False],
    )
    def test_collate_variable_tensors(self, layer_thick: bool) -> None:
        """Test collate_variable_tensors function."""
        data = [torch.randn(5, 3), torch.randn(5, 3), torch.randn(5, 3)]
        preprocessor = Mock()
        preprocessor.return_value = torch.randn(15, 3)

        inputs, targets = collate_variable_tensors(
            data, preprocessor, thick_deepest_layer=layer_thick, n_params=2
        )

        assert inputs.shape == (15, 2)
        assert targets.shape == (15,)

    def test_collate_variable_tensors_invalid_sizes(self) -> None:
        """
        Test collate_variable_tensors function fails
        when stacking different batch size tensors.
        """
        data = [torch.randn(5, 3), torch.randn(3, 3), torch.randn(7, 3)]
        preprocessor = Mock()
        preprocessor.return_value = torch.randn(15, 3)

        with pytest.raises(
            RuntimeError, match="stack expects each tensor to be equal size, but got"
        ):
            collate_variable_tensors(
                data, preprocessor, thick_deepest_layer=True, n_params=2
            )
