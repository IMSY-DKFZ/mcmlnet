"""Tests for mcmlnet.utils.convenience module."""

import os
import warnings
from collections.abc import Iterator
from typing import ClassVar
from unittest.mock import Mock

import lightning as pl
import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from mcmlnet.training.data_loading.preprocessing import PreProcessor
from mcmlnet.training.models.base_model import BaseModel
from mcmlnet.utils.convenience import (
    batch_inputs_to_three_layer_model,
    infer_model_device,
    load_trained_model,
    predict_in_batches,
    prepare_surrogate_model_data,
    run_model_from_physical_data,
    warn_out_of_range,
)


class RecordingLinearModel(pl.LightningModule):
    """Simple model that records forward-call metadata for batch tests."""

    def __init__(self, in_features: int = 3, out_features: int = 5) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(in_features, out_features)
        self.call_count = 0
        self.seen_devices: list[torch.device] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        self.seen_devices.append(x.device)
        return self.layer(x)


class OOMOnceModel(pl.LightningModule):
    """Model that raises OOM for batches larger than a threshold."""

    def __init__(self, max_batch_size: int) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        self.call_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        if len(x) > self.max_batch_size:
            raise torch.OutOfMemoryError("simulated oom")
        return torch.ones(len(x), 1, device=x.device)


class AlwaysOOMModel(pl.LightningModule):
    """Model that always raises an OOM error."""

    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        raise torch.OutOfMemoryError("simulated oom")


class BufferOnlyModel(pl.LightningModule):
    """Model that exposes a device only through registered buffers."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("buffer", torch.ones(1))


class NoStateModel:
    """Minimal model-like object without parameters, buffers, or device."""

    def parameters(self) -> Iterator[None]:
        return iter(())

    def buffers(self) -> Iterator[None]:
        return iter(())


class DeviceAttrModel(NoStateModel):
    """Minimal model-like object exposing only a device attribute."""

    def __init__(self, device: torch.device | str) -> None:
        self.device = device


def test_load_trained_model(caplog: pytest.LogCaptureFixture) -> None:
    """Test successful model loading of longest trained model (integration test)."""
    model, preprocessor, loaded_config = load_trained_model(
        os.path.join(
            os.environ["data_dir"], "models", "MLP_100M_photons_fold_0_tdr_6.0"
        ),
        "checkpoints/ForwardSurrogateModel-epoch=999-val_loss=0.0000.ckpt",
        BaseModel,
    )

    assert isinstance(model, BaseModel)
    assert isinstance(preprocessor, PreProcessor)
    assert isinstance(loaded_config, DictConfig)
    # Check most important model attributes
    assert hasattr(model, "cfg")
    assert model.cfg == loaded_config
    assert hasattr(model, "data_module")
    assert hasattr(model, "forward")
    assert model.training is False  # Model should be in eval mode
    for _, param in model.named_parameters():
        assert param.requires_grad is False  # All params should be frozen
        assert param.device.type == "cuda"
    # Check most important preprocessor and config attributes
    assert hasattr(preprocessor, "norm_1")
    assert hasattr(preprocessor, "norm_2")
    assert hasattr(loaded_config, "dataset")


def test_prepare_surrogate_model_data_success() -> None:
    """Test successful data preparation."""
    # Create mock preprocessor
    preprocessor = Mock(spec=PreProcessor)
    preprocessor.return_value = torch.randn(10, 15, 16)

    for thick_deepest_layer in [True, False]:
        # Create mock config
        config = Mock()
        config.dataset.thick_deepest_layer = thick_deepest_layer

        data = torch.randn(10, 5)

        result = prepare_surrogate_model_data(preprocessor, config, data)

        assert isinstance(result, torch.Tensor)
        if thick_deepest_layer:
            # Check that the second-to-last column was set to 0
            assert torch.all(result[..., -2] == 0)
        preprocessor.assert_called_once_with(data)
        preprocessor.reset_mock()


class TestInferModelDevice:
    """Direct tests for infer_model_device branch selection."""

    def test_prefers_parameter_device(self) -> None:
        """Parameters should take precedence over all other device hints."""
        model = RecordingLinearModel()

        assert infer_model_device(model, fallback="meta") == model.layer.weight.device

    def test_uses_buffer_device_when_no_parameters_exist(self) -> None:
        """Buffers should be used when the model has no parameters."""
        model = BufferOnlyModel()

        assert infer_model_device(model, fallback="meta") == model.buffer.device

    def test_uses_torch_device_attribute_when_present(self) -> None:
        """A torch.device attribute should be respected."""
        model = DeviceAttrModel(torch.device("cpu"))

        assert infer_model_device(model, fallback="meta") == torch.device("cpu")

    def test_uses_string_device_attribute_when_present(self) -> None:
        """A string device attribute should be converted to torch.device."""
        model = DeviceAttrModel("cpu")

        assert infer_model_device(model, fallback="meta") == torch.device("cpu")

    def test_returns_fallback_when_no_other_device_signal_exists(self) -> None:
        """Fallback should be used only when no device information exists."""
        model = NoStateModel()

        assert infer_model_device(model, fallback="cpu") == torch.device("cpu")


class TestPredictInBatches:
    """Test cases for predict_in_batches function."""

    def test_predict_in_batches_single_batch(self) -> None:
        """Test prediction with single batch."""
        model = RecordingLinearModel()
        data = torch.randn(10, 3)

        result = predict_in_batches(model, data, batch_size=-1)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (10, 5)
        assert result.device.type == "cpu"
        assert result.requires_grad is False
        assert model.call_count == 1

    def test_predict_in_batches_large_batch_size(self) -> None:
        """Test prediction with batch size larger than data."""
        model = RecordingLinearModel()
        data = torch.randn(10, 3)

        result = predict_in_batches(model, data, batch_size=20)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (10, 5)
        assert result.device.type == "cpu"
        assert model.call_count == 1

    def test_predict_in_batches_multiple_batches(self) -> None:
        """Test prediction with multiple batches."""
        model = RecordingLinearModel()
        data = torch.randn(10, 3)

        result = predict_in_batches(model, data, batch_size=5)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (10, 5)
        assert result.device.type == "cpu"
        assert model.call_count == 2

    def test_predict_in_batches_uses_model_device(self) -> None:
        """Test that input batches are moved to the model device."""
        model = RecordingLinearModel()
        data = torch.randn(10, 3)
        result = predict_in_batches(model, data, batch_size=5)

        assert isinstance(result, torch.Tensor)
        assert model.seen_devices
        assert all(device == model.layer.weight.device for device in model.seen_devices)

    def test_predict_in_batches_empty_data(self) -> None:
        """Test prediction with empty data."""
        model = RecordingLinearModel()

        with pytest.raises(ValueError, match="Input data must not be empty"):
            predict_in_batches(model, torch.empty(0, 3), batch_size=-1)

    def test_predict_in_batches_invalid_batch_size(self) -> None:
        """Test invalid batch sizes are rejected."""
        model = RecordingLinearModel()

        with pytest.raises(ValueError, match="Batch size must be at least 1 or -1"):
            predict_in_batches(model, torch.randn(4, 3), batch_size=0)

    def test_predict_in_batches_disables_autograd(self) -> None:
        """Test that prediction stays detached from autograd."""
        model = RecordingLinearModel()
        data = torch.randn(8, 3, requires_grad=True)

        result = predict_in_batches(model, data, batch_size=3)

        assert result.requires_grad is False
        assert result.device.type == "cpu"
        assert data.grad is None

    def test_predict_in_batches_progress_bar_alias(self) -> None:
        """Deprecated progress_bar alias should still control verbosity."""
        model = RecordingLinearModel()
        data = torch.randn(10, 3)

        result = predict_in_batches(model, data, batch_size=4, progress_bar=False)

        assert result.shape == (10, 5)
        assert model.call_count == 3

    def test_predict_in_batches_reduces_batch_size_on_oom(self) -> None:
        """OOM should trigger a retry with a smaller batch size."""
        model = OOMOnceModel(max_batch_size=2)
        data = torch.randn(4, 3)

        result = predict_in_batches(model, data, batch_size=4, verbose=False)

        assert result.shape == (4, 1)
        assert model.call_count == 3

    def test_predict_in_batches_oom_at_batch_size_one_raises(self) -> None:
        """Persistent OOM should fail once the batch size reaches one."""
        model = AlwaysOOMModel()
        data = torch.randn(3, 3)

        with pytest.raises(
            RuntimeError,
            match="Prediction failed with an out-of-memory error even for batch_size=1",
        ):
            predict_in_batches(model, data, batch_size=2, verbose=False)


class TestBatchInputsToThreeLayerModel:
    """Test cases for tissue model to three-layer model conversion."""

    def setup_method(self) -> None:
        """Setup for tests."""
        self.mu_a = np.array([1.0, 2.0])
        self.mu_s = np.array([10.0, 20.0])
        self.g = np.array([0.5, 0.6])
        self.n = np.array([1.33, 1.4])

    def test_single_layer_expands_to_three(self) -> None:
        """Test single-layer to three-layer expansion."""
        physical_params = batch_inputs_to_three_layer_model(
            self.mu_a, self.mu_s, self.g, self.n, d=None, dummy_thickness=0.0015
        )
        assert physical_params.shape == (2, 15)
        assert np.all(physical_params[:, 0:3] == self.mu_a[:, None])
        assert np.all(physical_params[:, 3:6] == self.mu_s[:, None])
        assert np.all(physical_params[:, 6:9] == self.g[:, None])
        assert np.all(physical_params[:, 9:12] == self.n[:, None])
        assert np.allclose(physical_params[:, 12:15], 0.0015)

    @pytest.mark.parametrize("d", [np.array([[0.0005, 0.0015]]), None])  # type: ignore[misc]
    def test_two_layers_pads_last_layer(self, d: np.ndarray | None) -> None:
        """Test two-layer to three-layer expansion."""
        physical_params = batch_inputs_to_three_layer_model(
            self.mu_a[None, :],
            self.mu_s[None, :],
            self.g[None, :],
            self.n[None, :],
            d,
            dummy_thickness=0.01,
        )
        # third layer repeats last layer, except thickness forced to dummy_thickness
        assert np.allclose(physical_params[0, 0:3], [1.0, 2.0, 2.0])
        assert np.allclose(physical_params[0, 3:6], [10.0, 20.0, 20.0])
        assert np.allclose(physical_params[0, 6:9], [0.5, 0.6, 0.6])
        assert np.allclose(physical_params[0, 9:12], [1.33, 1.4, 1.4])
        if d is None:
            np.allclose(physical_params[0, 12:15], [0.01, 0.01, 0.01])
        else:
            assert np.allclose(physical_params[0, 12:15], [0.0005, 0.0015, 0.01])

    def test_three_layers_passthrough(self) -> None:
        """Test that three-layer data is passed unchanged."""
        x = np.arange(3).reshape(1, 3)
        physical_params = batch_inputs_to_three_layer_model(x, x, x, x, x)

        for i in range(5):
            assert np.all(physical_params[:, i * 3 : (i + 1) * 3] == x)

    @pytest.mark.parametrize(  # type: ignore[misc]
        "data",
        [
            np.array(1.0),
            np.zeros((1, 2, 3)),
        ],
    )
    def test_invalid_data_dim_raises(self, data: np.ndarray) -> None:
        """Test that passing an invalid layer amount raises an error."""
        with pytest.raises(ValueError, match="Inputs must be 1D or 2D"):
            batch_inputs_to_three_layer_model(data, data, data, data)

    @pytest.mark.parametrize("n_layers", [0, 4])  # type: ignore[misc]
    def test_invalid_layer_count_raises(self, n_layers: int) -> None:
        """Test that passing an invalid layer amount raises an error."""
        data = np.zeros((1, n_layers))
        with pytest.raises(ValueError, match="Only 1, 2, or 3 layers are supported"):
            batch_inputs_to_three_layer_model(data, data, data, data)

    def test_shape_mismatch_raises(self) -> None:
        """Test that incompatible parameter shapes raise an error."""
        mu_a = np.zeros((1, 2))
        mu_s = np.zeros((1, 1))
        g = np.zeros((1, 2))
        n = np.zeros((1, 2))
        with pytest.raises(
            ValueError, match="mu_a, mu_s, g, n must share the same shape"
        ):
            batch_inputs_to_three_layer_model(mu_a, mu_s, g, n)

        mu_a = np.zeros((1, 2))
        mu_s = np.zeros((1, 2))
        g = np.zeros((1, 2))
        n = np.zeros((1, 2))
        d = np.zeros((1, 1))
        with pytest.raises(
            ValueError, match="d must share the same shape with the other parameters"
        ):
            batch_inputs_to_three_layer_model(mu_a, mu_s, g, n, d)


class TestWarnOutOfRange:
    """Direct tests for NumPy and Torch range warnings."""

    VALID_STACKED: ClassVar[list[float]] = [
        0.5,
        0.5,
        0.5,
        420.0,
        420.0,
        420.0,
        0.85,
        0.85,
        0.9,
        1.37,
        1.37,
        1.37,
        0.001,
        0.001,
        0.001,
    ]

    def test_numpy_input_emits_warning(self) -> None:
        """NumPy inputs outside the expected bounds should warn."""
        stacked = np.array([self.VALID_STACKED], dtype=np.float32)
        stacked[0, 8] = 0.99

        with pytest.warns(UserWarning, match=r"g outside \[0.8, 0.95\]"):
            warn_out_of_range(stacked)

    def test_torch_input_emits_warning(self) -> None:
        """Torch inputs outside the expected bounds should warn."""
        stacked = torch.tensor([self.VALID_STACKED], dtype=torch.float32)
        stacked[0, 8] = 0.3

        with pytest.warns(UserWarning, match=r"g outside \[0.8, 0.95\]"):
            warn_out_of_range(stacked)

    @pytest.mark.parametrize(
        "stacked",
        [  # type: ignore[misc]
            np.array([VALID_STACKED], dtype=np.float32),
            torch.tensor([VALID_STACKED], dtype=torch.float32),
        ],
    )
    def test_in_range_input_does_not_warn(
        self, stacked: np.ndarray | torch.Tensor
    ) -> None:
        """Inputs within bounds should stay silent for both array types."""
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            warn_out_of_range(stacked)

        assert not record


class TestRunModelFromPhysicalData:
    """Basic test cases for simple model inference interface."""

    def setup_method(self) -> None:
        """Setup for tests."""
        self.mu_a = np.full((1, 3), 0.5, dtype=np.float32)
        self.mu_s = np.full((1, 3), 420.0, dtype=np.float32)
        self.g = np.full((1, 3), 0.85, dtype=np.float32)
        self.n = np.full((1, 3), 1.37, dtype=np.float32)
        self.d = np.full((1, 3), 0.001, dtype=np.float32)

    @pytest.mark.parametrize("specular", [False, True])  # type: ignore[misc]
    def test_end_to_end_inference(self, specular: bool) -> None:
        """Basic end-to-end test."""
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            reflectances = run_model_from_physical_data(
                self.mu_a,
                self.mu_s,
                self.g,
                self.n,
                self.d,
                add_specular_reflectance=specular,
                inference_batch_size=2,
            )

        assert isinstance(reflectances, np.ndarray)
        assert reflectances.shape == (self.mu_a.shape[0], 1)
        assert np.isfinite(reflectances).all()

        assert not record  # no warnings expected

    def test_warn_out_of_range_emits_warning(self) -> None:
        """Test that end-to-end test warns if parameters are out of (training) range."""
        with pytest.warns(UserWarning, match=r"g outside \[0.8, 0.95\]"):
            reflectances = run_model_from_physical_data(
                self.mu_a,
                self.mu_s,
                np.array([[0.8, 0.85, 0.99]]),
                self.n,
                self.d,
                add_specular_reflectance=False,
                inference_batch_size=2,
            )
        assert isinstance(reflectances, np.ndarray)
        assert reflectances.shape == (self.mu_a.shape[0], 1)

        with pytest.warns(UserWarning, match=r"g outside \[0.8, 0.95\]"):
            reflectances = run_model_from_physical_data(
                self.mu_a,
                self.mu_s,
                np.array([[0.3, 0.85, 0.92]]),
                self.n,
                self.d,
                add_specular_reflectance=False,
                inference_batch_size=2,
            )
        assert isinstance(reflectances, np.ndarray)
        assert reflectances.shape == (self.mu_a.shape[0], 1)
