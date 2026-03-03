"""Tests for mcmlnet.utils.rescaling module."""

import copy

import numpy as np
import pytest
import torch

from mcmlnet.utils.rescaling import DataRescaler


class TestDataRescaler:
    """Test cases for DataRescaler class."""

    def setup_method(self) -> None:
        """Setup common variables for tests."""
        self.data_2d = torch.randn(100, 5)
        self.data_3d = torch.randn(100, 5, 3)
        self.data_np = np.random.randn(100, 5)
        self.norm_1 = torch.tensor([1.0, 2.0])
        self.norm_2 = torch.tensor([0.5, 1.0])

    @pytest.mark.parametrize(  # type: ignore[misc]
        "scale", ["min-max", "min-max-sym", "z-score", "None", None]
    )
    def test_init_valid_scales(self, scale: str | None) -> None:
        rescaler = DataRescaler(scale=scale)
        assert rescaler.scale == scale
        assert rescaler.eps == 1e-8
        assert rescaler.norm_1 is None
        assert rescaler.norm_2 is None

    @pytest.mark.parametrize("scale", ["invalid_scale", 123, object()])  # type: ignore[misc]
    def test_init_invalid_scale(self, scale: str | int | object) -> None:
        """Test initialization with invalid scale parameter."""
        with pytest.raises(ValueError, match="Invalid scale"):
            DataRescaler(scale=scale)  # type: ignore[arg-type]

    def test_init_with_custom_eps(self) -> None:
        """Test initialization with custom epsilon value."""
        rescaler = DataRescaler(scale="z-score", eps=1e-6)
        assert rescaler.eps == 1e-6

    def test_init_with_norm_tensors(self) -> None:
        """Test initialization with provided normalization tensors."""
        rescaler = DataRescaler(scale="z-score", norm_1=self.norm_1, norm_2=self.norm_2)
        assert torch.equal(rescaler.norm_1, self.norm_1)
        assert torch.equal(rescaler.norm_2, self.norm_2)

    @pytest.mark.parametrize(  # type: ignore[misc]
        "data",
        [
            pytest.param(torch.randn(10, 5), id="2d"),
            pytest.param(torch.randn(10, 5, 3), id="3d"),
        ],
    )
    def test_validate_inputs_valid_tensor(self, data: torch.Tensor) -> None:
        """Test validation of valid input data."""
        DataRescaler._validate_inputs(data)

    @pytest.mark.parametrize(  # type: ignore[misc]
        "data,err,msg",
        [
            (np.random.randn(10, 5), TypeError, r"Data must be a torch.Tensor"),
            (torch.randn(10), ValueError, "Data must be 2D or 3D"),
            (torch.randn(10, 5, 3, 2), ValueError, "Data must be 2D or 3D"),
        ],
    )
    def test_validate_inputs_invalid(
        self, data: np.ndarray | torch.Tensor, err: Exception, msg: str
    ) -> None:
        """Test validation of input data."""
        with pytest.raises(err, match=msg):
            DataRescaler._validate_inputs(data)

    @pytest.mark.parametrize(  # type: ignore[misc]
        "reference_ids,msg",
        [
            ([15], "Reference indices out of bounds"),
            ([], "Reference indices cannot be empty"),
        ],
    )
    def test_validate_inputs_invalid_reference_ids(
        self, reference_ids: list[int], msg: str
    ) -> None:
        """Test validation of reference indices."""
        data = torch.randn(10, 5)
        with pytest.raises(ValueError, match=msg):
            DataRescaler._validate_inputs(data, reference_ids=reference_ids)

    @pytest.mark.parametrize(  # type: ignore[misc]
        "data,expected",
        [
            (torch.randn(10, 5), (None, slice(None))),
            (torch.randn(10, 5, 3), (None, None, slice(None))),
        ],
    )
    def test_expand_dims(self, data: torch.Tensor, expected: tuple | None) -> None:
        """Test dimension expansion based on data shape."""
        assert DataRescaler._expand_dims(data) == expected

    @pytest.mark.parametrize(  # type: ignore[misc]
        "data,expected",
        [
            (torch.randn(10, 5), 0),
            (torch.randn(10, 5, 3), (0, 1)),
        ],
    )
    def test_dims(self, data: torch.Tensor, expected: int | tuple) -> None:
        """Test dimension extraction based on data shape."""
        assert DataRescaler._dims(data) == expected

    @pytest.mark.parametrize("scale", [None, "None"])  # type: ignore[misc]
    def test_check_scaling_params_no_scale(self, scale: str | None) -> None:
        """Test that checking scaling parameters passes when no scaling is applied."""
        rescaler = DataRescaler(scale=scale)
        rescaler._check_scaling_params()  # Should not raise

    @pytest.mark.parametrize("scale", ["z-score", "min-max", "min-max-sym"])  # type: ignore[misc]
    def test_check_scaling_params_missing_params(self, scale: str) -> None:
        """Test that checking scaling parameters raises error if not set."""
        rescaler = DataRescaler(scale=scale)
        with pytest.raises(RuntimeError, match="Scaling parameters not computed"):
            rescaler._check_scaling_params()

    @pytest.mark.parametrize(  # type: ignore[misc]
        "scale,data,shape",
        [
            ("z-score", "data_2d", (5,)),
            ("z-score", "data_3d", (3,)),
            ("min-max", "data_2d", (5,)),
            ("min-max", "data_3d", (3,)),
            ("min-max-sym", "data_2d", (5,)),
            ("min-max-sym", "data_3d", (3,)),
        ],
    )
    def test_fit(self, scale: str, data: str, shape: tuple) -> None:
        """Test fitting the rescaler."""
        rescaler = DataRescaler(scale=scale)
        result = rescaler.fit(getattr(self, data))
        assert result is rescaler
        assert rescaler.norm_1 is not None
        assert rescaler.norm_2 is not None
        assert rescaler.norm_1.shape == shape
        assert rescaler.norm_2.shape == shape

    @pytest.mark.parametrize("data", ["data_2d", "data_3d"])  # type: ignore[misc]
    def test_fit_with_reference_ids(self, data: str) -> None:
        """Test fitting with specific reference indices."""
        rescaler = DataRescaler(scale="z-score")
        reference_ids = [0, 1, 2, 3, 4]  # Use first 5 samples to fit the rescaler

        result = rescaler.fit(getattr(self, data).clone(), reference_ids=reference_ids)

        assert result is rescaler
        # Check that only reference data was used for fitting
        reference_data = getattr(self, data).clone()[reference_ids]
        if reference_data.ndim == 2:
            expected_mean = reference_data.mean(dim=0)
        elif reference_data.ndim == 3:
            expected_mean = reference_data.mean(dim=(0, 1))
        else:
            raise ValueError("Unexpected tensor shape")
        assert torch.allclose(rescaler.norm_1, expected_mean, atol=1e-6)

    @pytest.mark.parametrize(  # type: ignore[misc]
        "scale,data",
        [
            (None, "data_2d"),
            (None, "data_3d"),
            ("None", "data_2d"),
            ("None", "data_3d"),
        ],
    )
    def test_fit_no_scale(self, scale: str | None, data: str) -> None:
        """Test fitting when no scaling is applied."""
        rescaler = DataRescaler(scale=scale)

        result = rescaler.fit(getattr(self, data).clone())

        assert result is rescaler
        assert rescaler.norm_1 is None
        assert rescaler.norm_2 is None

    def test_transform_z_score(self) -> None:
        """Test z-score transformation."""
        # Larger dataset for more reliable stat.s
        for manual_data in [torch.randn(1000, 5), torch.randn(1000, 3, 5)]:
            rescaler = DataRescaler(scale="z-score")
            rescaler.fit(manual_data)

            transformed, norm_1, norm_2 = rescaler.transform(manual_data)

            assert isinstance(transformed, torch.Tensor)
            assert torch.equal(norm_1, rescaler.norm_1)
            assert torch.equal(norm_2, rescaler.norm_2)

            # Check that transformed data has mean close to 0 and std close to 1
            if manual_data.ndim == 2:
                mean, std = transformed.mean(dim=0), transformed.std(dim=0)
            elif manual_data.ndim == 3:
                mean, std = transformed.mean(dim=(0, 1)), transformed.std(dim=(0, 1))
            else:
                raise ValueError("Unexpected tensor shape")
            assert torch.allclose(mean, torch.zeros(5), atol=1e-6)
            assert torch.allclose(std, torch.ones(5), atol=1e-6)

    @pytest.mark.parametrize("data", ["data_2d", "data_3d"])  # type: ignore[misc]
    def test_transform_min_max(self, data: str) -> None:
        """Test min-max transformation."""
        rescaler = DataRescaler(scale="min-max")
        rescaler.fit(getattr(self, data).clone())

        transformed, norm_1, norm_2 = rescaler.transform(getattr(self, data).clone())

        assert isinstance(transformed, torch.Tensor)
        assert torch.equal(norm_1, rescaler.norm_1)
        assert torch.equal(norm_2, rescaler.norm_2)

        # Check that transformed data is in [0, 1] range
        assert torch.all(transformed >= 0)
        assert torch.all(transformed <= 1)
        assert torch.min(transformed) == 0
        assert torch.max(transformed) == 1

    @pytest.mark.parametrize("data", ["data_2d", "data_3d"])  # type: ignore[misc]
    def test_transform_min_max_sym(self, data: str) -> None:
        """Test min-max-sym transformation."""
        rescaler = DataRescaler(scale="min-max-sym")
        rescaler.fit(getattr(self, data).clone())

        transformed, norm_1, norm_2 = rescaler.transform(getattr(self, data).clone())

        assert isinstance(transformed, torch.Tensor)
        assert torch.equal(norm_1, rescaler.norm_1)
        assert torch.equal(norm_2, rescaler.norm_2)

        # Check that transformed data is in [-1, 1] range
        assert torch.all(transformed >= -1)
        assert torch.all(transformed <= 1)
        assert torch.min(transformed) == -1
        assert torch.max(transformed) == 1

    @pytest.mark.parametrize("data", ["data_2d", "data_3d"])  # type: ignore[misc]
    def test_fit_transform(self, data: str) -> None:
        """Test fit_transform method."""
        rescaler = DataRescaler(scale="z-score")

        transformed, norm_1, norm_2 = rescaler.fit_transform(
            getattr(self, data).clone()
        )

        assert isinstance(transformed, torch.Tensor)
        assert isinstance(norm_1, torch.Tensor)
        assert isinstance(norm_2, torch.Tensor)
        assert torch.equal(norm_1, rescaler.norm_1)
        assert torch.equal(norm_2, rescaler.norm_2)
        if getattr(self, data).ndim == 2:
            mean, std = transformed.mean(dim=0), transformed.std(dim=0)
            assert torch.allclose(mean, torch.zeros(5), atol=1e-6)
            assert torch.allclose(std, torch.ones(5), atol=1e-6)
        elif getattr(self, data).ndim == 3:
            mean, std = transformed.mean(dim=(0, 1)), transformed.std(dim=(0, 1))
            assert torch.allclose(mean, torch.zeros(3), atol=1e-6)
            assert torch.allclose(std, torch.ones(3), atol=1e-6)
        else:
            raise ValueError("Unexpected tensor shape")

    @pytest.mark.parametrize("data", ["data_2d", "data_3d"])  # type: ignore[misc]
    def test_fit_transform_min_max(self, data: str) -> None:
        """Test fit_transform method with min-max scaling."""
        rescaler = DataRescaler(scale="min-max")
        transformed, norm_1, norm_2 = rescaler.fit_transform(
            getattr(self, data).clone()
        )

        assert isinstance(transformed, torch.Tensor)
        assert isinstance(norm_1, torch.Tensor)
        assert isinstance(norm_2, torch.Tensor)
        assert torch.equal(norm_1, rescaler.norm_1)
        assert torch.equal(norm_2, rescaler.norm_2)
        assert torch.all(transformed >= 0)
        assert torch.all(transformed <= 1)
        assert torch.min(transformed) == 0
        assert torch.max(transformed) == 1

    @pytest.mark.parametrize("data", ["data_2d", "data_3d"])  # type: ignore[misc]
    def test_fit_transform_min_max_sym(self, data: str) -> None:
        """Test fit_transform method with min-max-sym scaling."""
        rescaler = DataRescaler(scale="min-max-sym")
        transformed, norm_1, norm_2 = rescaler.fit_transform(
            getattr(self, data).clone()
        )

        assert isinstance(transformed, torch.Tensor)
        assert isinstance(norm_1, torch.Tensor)
        assert isinstance(norm_2, torch.Tensor)
        assert torch.equal(norm_1, rescaler.norm_1)
        assert torch.equal(norm_2, rescaler.norm_2)
        assert torch.all(transformed >= -1)
        assert torch.all(transformed <= 1)
        assert torch.min(transformed) == -1
        assert torch.max(transformed) == 1

    @pytest.mark.parametrize(  # type: ignore[misc]
        "scale,data",
        [
            ("z-score", "data_2d"),
            ("z-score", "data_3d"),
            ("min-max", "data_2d"),
            ("min-max", "data_3d"),
            ("min-max-sym", "data_2d"),
            ("min-max-sym", "data_3d"),
        ],
    )
    def test_call_method(
        self,
        scale: str,
        data: str,
    ) -> None:
        """Test __call__ method."""
        rescaler = DataRescaler(scale=scale)
        rescaler.fit(getattr(self, data))

        result = rescaler(getattr(self, data))

        assert isinstance(result, torch.Tensor)
        assert result.shape == getattr(self, data).shape

    @pytest.mark.parametrize(  # type: ignore[misc]
        "scale,data",
        [
            (None, "data_2d"),
            (None, "data_3d"),
            ("None", "data_2d"),
            ("None", "data_3d"),
        ],
    )
    def test_call_method_no_scale(self, scale: str | None, data: str) -> None:
        """Test __call__ method with no scaling."""
        rescaler = DataRescaler(scale=scale)

        result = rescaler(getattr(self, data).clone())

        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, getattr(self, data).clone())

    @pytest.mark.parametrize(  # type: ignore[misc]
        "scale,data",
        [
            ("z-score", "data_2d"),
            ("z-score", "data_3d"),
            ("min-max", "data_2d"),
            ("min-max", "data_3d"),
            ("min-max-sym", "data_2d"),
            ("min-max-sym", "data_3d"),
        ],
    )
    def test_inverse_transform_z_score(self, scale: str, data: str) -> None:
        """Test inverse transformation for scaling."""
        rescaler = DataRescaler(scale=scale)
        rescaler.fit(getattr(self, data).clone())

        transformed = rescaler(getattr(self, data).clone())
        inverse_transformed = rescaler.inverse_transform(transformed)

        assert torch.allclose(
            inverse_transformed, getattr(self, data).clone(), atol=1e-6
        )

    @pytest.mark.parametrize(  # type: ignore[misc]
        "scale,data",
        [
            (None, "data_2d"),
            (None, "data_3d"),
            ("None", "data_2d"),
            ("None", "data_3d"),
        ],
    )
    def test_inverse_transform_no_scale(self, scale: str | None, data: str) -> None:
        """Test inverse transformation when no scaling is applied."""
        rescaler = DataRescaler(scale=scale)
        result = rescaler.inverse_transform(getattr(self, data).clone())
        assert torch.equal(result, getattr(self, data).clone())

    @pytest.mark.parametrize("data", ["data_2d", "data_3d"])  # type: ignore[misc]
    def test_inverse_transform_not_implemented(self, data: str) -> None:
        """Test inverse transformation with unsupported scale method."""
        rescaler = DataRescaler(scale=None)
        rescaler.scale = (
            "invalid_scale"  # usually, we do not end up here, but just in case
        )
        rescaler.norm_1 = torch.tensor([1.0])
        rescaler.norm_2 = torch.tensor([1.0])

        with pytest.raises(ValueError, match="Invalid scale 'invalid_scale'"):
            rescaler.inverse_transform(getattr(self, data).clone())

    @pytest.mark.parametrize("scale", ["z-score", "min-max", "min-max-sym"])  # type: ignore[misc]
    def test_device_handling(self, scale: str) -> None:
        """Test that rescaling works correctly with different devices."""
        if torch.cuda.is_available():
            data_cpu = self.data_2d.clone()
            data_gpu = data_cpu.cuda()

            rescaler = DataRescaler(scale=scale)
            rescaler.fit(data_cpu)

            # Test transformation on GPU data
            result_gpu = rescaler(data_gpu)
            assert result_gpu.device == data_gpu.device

            # Test inverse transformation on GPU data
            inverse_gpu = rescaler.inverse_transform(result_gpu)
            assert inverse_gpu.device == data_gpu.device

    def test_caching_behavior(self) -> None:
        """Test that caching works correctly."""
        data = torch.randn(100, 5)
        rescaler = DataRescaler(scale="z-score")
        rescaler.fit(data)

        # First call should populate cache
        rescaler(data)
        assert len(rescaler._cached_norms) > 0
        assert len(rescaler._cached_eps) > 0

    def test_cache_cleanup(self) -> None:
        """Test that old cache entries are cleaned up."""
        data = torch.randn(100, 5)
        rescaler = DataRescaler(scale="z-score")
        rescaler.fit(data)

        # Create multiple cache entries
        for i in range(6):  # More than the cleanup threshold of 4
            device_data = data.to(f"cpu:{i}" if i > 0 else "cpu")
            rescaler(device_data)

        # Check that cache size is limited
        assert len(rescaler._cached_norms) <= 4

    @pytest.mark.parametrize("scale", ["z-score", "min-max", "min-max-sym"])  # type: ignore[misc]
    def test_small_std_handling(self, scale: str) -> None:
        """Test handling of very small standard deviations."""
        # Create data with very small std in one dimension
        data = torch.randn(100, 5)
        data[:, 0] = 1.0  # Constant column
        data[:, 1] = data[:, 1] * 1e-10  # Very small std

        rescaler = DataRescaler(scale=scale, eps=1e-8)
        rescaler.fit(data)

        result = rescaler(data)

        # Should not have any infinite or NaN values
        assert torch.all(torch.isfinite(result))

    @pytest.mark.parametrize("scale", ["z-score", "min-max", "min-max-sym"])  # type: ignore[misc]
    def test_fit_rejects_numpy_input(self, scale: str) -> None:
        """Test that fit method rejects numpy input (since it's not decorated)."""
        rescaler = DataRescaler(scale=scale)

        with pytest.raises(TypeError, match=r"Data must be a torch.Tensor"):
            rescaler.fit(copy.deepcopy(self.data_np))

    @pytest.mark.parametrize("scale", ["z-score", "min-max", "min-max-sym"])  # type: ignore[misc]
    def test_transform_rejects_numpy_input(self, scale: str) -> None:
        """Test that transform method rejects numpy input (since it's not decorated)."""
        rescaler = DataRescaler(
            scale=scale, norm_1=torch.randn(5), norm_2=torch.randn(5)
        )

        with pytest.raises(TypeError, match=r"Data must be a torch.Tensor"):
            rescaler.transform(copy.deepcopy(self.data_np))

    @pytest.mark.parametrize("scale", ["z-score", "min-max", "min-max-sym"])  # type: ignore[misc]
    def test_fit_transform_accepts_numpy_input(self, scale: str) -> None:
        """Test that fit_transform method accepts numpy input (since it's decorated)."""
        rescaler = DataRescaler(scale=scale)

        result = rescaler.fit_transform(copy.deepcopy(self.data_np))

        assert isinstance(result[0], np.ndarray)
        assert result[0].shape == self.data_np.shape

    @pytest.mark.parametrize("scale", ["z-score", "min-max", "min-max-sym"])  # type: ignore[misc]
    def test_edge_case_empty_data(self, scale: str) -> None:
        """Test edge case with empty data."""
        data = torch.empty(0, 5)
        rescaler = DataRescaler(scale=scale)

        with pytest.raises(ValueError):
            rescaler.fit(data)

    def test_edge_case_single_sample(self) -> None:
        """Test z-score edge case with single sample."""
        data = torch.randn(1, 5)
        rescaler = DataRescaler(scale="z-score")

        # Suppress the warning about degrees of freedom
        with pytest.warns(UserWarning, match="std\\(\\): degrees of freedom is <= 0"):
            # Should work but may have issues with std calculation
            result = rescaler.fit_transform(data)
            assert isinstance(result[0], torch.Tensor)
