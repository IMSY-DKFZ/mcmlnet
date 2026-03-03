"""Tests for mcmlnet.utils.network_init module."""

import copy

import pytest
import torch
import torch.nn as nn

from mcmlnet.utils.network_init import (
    init_weights,
    matmul_precision,
    set_activation,
)


class TestMatmulPrecision:
    """Test cases for matmul_precision function."""

    @pytest.mark.parametrize(  # type: ignore[misc]
        "precision", ["bf16-mixed", "bf16", "float64", "double", "float32", "fp32"]
    )
    def test_matmul_precision(self, precision: str) -> None:
        """
        Test setting various precisions.

        NOTE: We cannot directly test the internal state of PyTorch's matmul precision,
        but we can ensure that the function runs without errors for valid inputs.
        """
        matmul_precision(precision)


class TestInitWeights:
    """Test cases for init_weights function."""

    def setup_method(self) -> None:
        """Setup method to initialize common variables."""
        torch.manual_seed(42)  # Ensure reproducibility for tests
        self.layer = nn.Linear(10, 5)
        self.activation = nn.ReLU()

    @pytest.mark.parametrize(  # type: ignore[misc]
        "module, init_method, activation",
        [
            (module, init_method, activation)
            for module in [
                nn.Linear(10, 5),
                nn.Conv1d(3, 6, kernel_size=3),
                nn.Conv2d(3, 6, kernel_size=3),
                nn.Conv3d(3, 6, kernel_size=3),
                nn.ConvTranspose2d(3, 6, kernel_size=3),
            ]
            for init_method in [
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                "orthogonal",
                "sparse",
            ]
            for activation in ["relu", "leaky_relu", "selu", "tanh"]
            if not (init_method == "sparse" and not isinstance(module, nn.Linear))
        ],
    )
    def test_layer_weight_initialization(
        self, module: nn.Module, init_method: str, activation: str
    ) -> None:
        """Test weight initialization for Linear layer with various methods."""
        if init_method == "sparse" and not isinstance(module, nn.Linear):
            pytest.skip("Sparse initialization is only applicable to Linear layers.")
        init_weights(module, weight_init=init_method, activation=activation)

        assert not torch.allclose(module.weight, torch.zeros_like(module.weight))
        assert torch.allclose(module.bias, torch.zeros_like(module.bias))

    @pytest.mark.parametrize(  # type: ignore[misc]
        "module",
        [
            nn.Linear(10, 5),
            nn.Conv1d(3, 6, kernel_size=3),
            nn.Conv2d(3, 6, kernel_size=3),
            nn.Conv3d(3, 6, kernel_size=3),
            nn.ConvTranspose2d(3, 6, kernel_size=3),
        ],
    )
    def test_layer_bias_ones(self, module: nn.Module) -> None:
        """Test bias initialization with ones."""
        init_weights(module, bias_init="ones")

        assert torch.allclose(module.bias, torch.ones_like(module.bias))

    @pytest.mark.parametrize(  # type: ignore[misc]
        "module",
        [
            nn.Linear(10, 5),
            nn.Conv1d(3, 6, kernel_size=3),
            nn.Conv2d(3, 6, kernel_size=3),
            nn.Conv3d(3, 6, kernel_size=3),
            nn.ConvTranspose2d(3, 6, kernel_size=3),
        ],
    )
    def test_layer_bias_constant(self, module: nn.Module) -> None:
        """Test bias initialization with constant."""
        init_weights(module, bias_init="constant")

        assert torch.allclose(module.bias, torch.full_like(module.bias, 0.01))

    @pytest.mark.parametrize(  # type: ignore[misc]
        "normalization_layer",
        [nn.BatchNorm1d(10), nn.BatchNorm2d(10), nn.BatchNorm3d(10)],
    )
    def test_batch_norm_zero_init(self, normalization_layer: nn.Module) -> None:
        """Test BatchNorm initialization with zero initialization."""
        init_weights(normalization_layer, zero_batchnorm=True)

        assert torch.allclose(
            normalization_layer.weight, torch.zeros_like(normalization_layer.weight)
        )
        assert torch.allclose(
            normalization_layer.bias, torch.zeros_like(normalization_layer.bias)
        )

    @pytest.mark.parametrize(  # type: ignore[misc]
        "normalization_layer",
        [
            nn.BatchNorm1d(10),
            nn.BatchNorm2d(10),
            nn.BatchNorm3d(10),
            nn.LayerNorm(10),
            nn.GroupNorm(2, 10),
        ],
    )
    def test_norm_layer_default_init(self, normalization_layer: nn.Module) -> None:
        """Test normalization layer initialization with default initialization."""
        init_weights(normalization_layer, zero_batchnorm=False)

        assert torch.allclose(
            normalization_layer.weight, torch.ones_like(normalization_layer.weight)
        )
        assert torch.allclose(
            normalization_layer.bias, torch.zeros_like(normalization_layer.bias)
        )

    @pytest.mark.parametrize(  # type: ignore[misc]
        "module",
        [
            nn.LSTM(10, 5),
            nn.GRU(10, 5),
        ],
    )
    def test_recurrent_layer(self, module: nn.Module) -> None:
        """Test recurrent layer initialization."""
        init_weights(module)

        for name, param in module.named_parameters():
            if "weight" in name:
                assert not torch.allclose(param, torch.zeros_like(param))
            elif "bias" in name:
                assert torch.allclose(param, torch.zeros_like(param))

    def test_embedding_layer(self) -> None:
        """Test Embedding layer initialization."""
        layer = nn.Embedding(100, 10)
        init_weights(layer)

        assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))

    def test_activation_layers(self) -> None:
        """Test that activation layers don't raise errors."""
        activations = [nn.ReLU(), nn.LeakyReLU(), nn.SELU(), nn.Tanh(), nn.Sigmoid()]

        for activation in activations:
            init_weights(activation)

    def test_sequential_layer(self) -> None:
        """Test Sequential layer initialization."""
        sequential_net = nn.Sequential(self.layer, nn.ReLU())
        sequential_net.apply(lambda module: init_weights(module))

        # Check that the contained Linear layer inside is initialized
        linear_layer = sequential_net[0]
        assert not torch.allclose(
            linear_layer.weight, torch.zeros_like(linear_layer.weight)
        )
        assert torch.allclose(linear_layer.bias, torch.zeros_like(linear_layer.bias))

    def test_invalid_activation(self) -> None:
        """Test initialization with invalid activation."""
        with pytest.raises(ValueError, match="Activation 'invalid' not supported"):
            init_weights(self.layer, activation="invalid")

    def test_invalid_weight_init(self) -> None:
        """Test initialization with invalid weight initialization."""
        with pytest.raises(
            ValueError, match="Weight initialization 'invalid' not supported"
        ):
            init_weights(self.layer, weight_init="invalid")

    def test_invalid_bias_init(self) -> None:
        """Test initialization with invalid bias initialization."""
        with pytest.raises(
            ValueError, match="Bias initialization 'invalid' not supported"
        ):
            init_weights(self.layer, bias_init="invalid")

    def test_negative_slope_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test warning for negative slope parameter."""
        caplog.clear()
        caplog.set_level("WARNING")

        init_weights(self.layer, activation="leaky_relu", negative_slope=-0.1)

        assert "slope' parameter should have positive value" in caplog.text

    def test_seed_reproducibility(self) -> None:
        """Test that seed ensures reproducible initialization."""
        layer1 = nn.Linear(10, 5)
        layer2 = nn.Linear(10, 5)

        # Initialize with same seed
        init_weights(layer1, seed=42)
        init_weights(layer2, seed=42)

        assert torch.allclose(layer1.weight, layer2.weight)
        assert torch.allclose(layer1.bias, layer2.bias)

    def test_different_seeds_different_weights(self) -> None:
        """Test that different seeds produce different weights."""
        layer1 = nn.Linear(10, 5)
        layer2 = copy.deepcopy(layer1)

        # Initialize copy of layer with different seeds
        init_weights(layer1, seed=42)
        init_weights(layer2, seed=123)

        assert not torch.allclose(layer1.weight, layer2.weight)

    def test_unknown_module_type(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test handling of unknown module types."""

        class UnknownModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.randn(10, 5))

        layer = UnknownModule()

        caplog.clear()
        caplog.set_level("INFO")

        init_weights(layer)

        assert "not initialized due to missing init method" in caplog.text


class TestSetActivation:
    """Test cases for set_activation function."""

    @pytest.mark.parametrize(  # type: ignore[misc]
        "activation_type, activation, negative_slope",
        [
            (nn.ReLU, "relu", None),
            (nn.LeakyReLU, "leaky_relu", 0.01),
            (nn.SELU, "selu", None),
            (nn.Tanh, "tanh", None),
        ],
    )
    def test_set_activation(
        self, activation_type: nn.Module, activation: str, negative_slope: float | None
    ) -> None:
        """Test setting activation."""
        network = nn.Sequential(
            nn.Linear(10, 5),
            nn.LeakyReLU(),
            nn.ReLU(),
            nn.Linear(5, 1),
            nn.SELU(),
            nn.Tanh(),
        )

        result = set_activation(network, activation, negative_slope=negative_slope)

        for layer_id in [1, 2, 4, 5]:
            assert isinstance(result[layer_id], activation_type)
            if negative_slope is not None:
                assert result[layer_id].negative_slope == negative_slope

    @pytest.mark.parametrize(  # type: ignore[misc]
        "activation, negative_slope",
        [
            ("relu", None),
            ("leaky_relu", 0.01),
            ("selu", None),
            ("tanh", None),
        ],
    )
    def test_set_activation_preserves_non_activation_layers(
        self, activation: str, negative_slope: float | None
    ) -> None:
        """Test that non-activation layers are preserved."""
        network = nn.Sequential(
            nn.Linear(10, 5),
            nn.LeakyReLU(),
            nn.ReLU(),
            nn.Linear(5, 1),
            nn.Sigmoid(),
            nn.SELU(),
            nn.Tanh(),
        )

        result = set_activation(network, activation, negative_slope=negative_slope)

        # Linear layers should be unchanged
        for layer_id in [0, 3]:
            assert isinstance(result[layer_id], nn.Linear)
        # Sigmoid layer should be unchanged
        assert isinstance(result[4], nn.Sigmoid)

    def test_set_activation_invalid_type(self) -> None:
        """Test setting invalid activation type."""
        network = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))

        with pytest.raises(ValueError, match="Activation 'invalid' not supported"):
            set_activation(network, "invalid")

    def test_set_activation_empty_network(self) -> None:
        """Test setting activation on empty network."""
        network = nn.Sequential()

        result = set_activation(network, "relu")

        assert result is network

    def test_set_activation_no_activation_layers(self) -> None:
        """Test setting activation on network with no activation layers."""
        network = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 1))

        result = set_activation(network, "relu")

        assert result is network

    def test_set_activation_default_negative_slope(self) -> None:
        """Test setting LeakyReLU with default negative slope."""
        network = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))

        result = set_activation(network, "leaky_relu")

        assert isinstance(result[1], nn.LeakyReLU)
        assert result[1].negative_slope == 0.0  # Default value
