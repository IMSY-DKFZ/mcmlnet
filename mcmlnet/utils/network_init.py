"""Network initialization utilities."""

from typing import Any

import torch
import torch.nn as nn

from mcmlnet.utils.logging import setup_logging

logger = setup_logging(level="info", logger_name=__name__)


def matmul_precision(precision: str) -> None:
    """
    Set the precision for matrix multiplication based on the given precision string.

    Args:
        precision: A string indicating the desired precision.
            Options include 'bf16-mixed', 'bf16', '32', '64', '16', '16-mixed'.
    """
    if precision == "bf16-mixed":
        torch.set_float32_matmul_precision(precision="high")
    elif precision == "bf16":
        torch.set_float32_matmul_precision(precision="medium")
    elif "64" in precision:
        pass
    else:
        torch.set_float32_matmul_precision(precision="highest")


def init_weights(
    module: nn.Module,
    activation: str = "leaky_relu",
    negative_slope: float = 0.0,
    zero_batchnorm: bool = False,
    weight_init: str = "xavier_uniform",
    bias_init: str = "zeros",
    seed: int = 42,
) -> None:
    """Initializes different layer type weights & biases with hard-coded, fixed seed.

    Args:
        module: Module with weights and/or biases.
        zero_batchnorm: Whether to zero initialize BatchNorm layers.
        activation: Activation layer type.
        negative_slope: Only required for leakyReLU activation,
            gives steepness of negative slope.
        weight_init: Initialization method of the nn.Module weights.
        bias_init: Initialization method of the nn.Module bias.
        seed: Random seed for reproducibility.

    Raises:
        ValueError: If the activation function, weight, or
            bias initialization is not supported.
    """
    # Set seed for reproducibility
    # Ensure that different layers are initialized with different seeds!
    torch.manual_seed(seed)

    # Parameter assertions and warnings
    supported_activations = ["relu", "leaky_relu", "selu", "tanh"]
    if activation not in supported_activations:
        raise ValueError(
            f"Activation '{activation}' not supported. "
            f"Use one of {supported_activations}."
        )

    # Warn if leaky ReLU slope value is badly chosen
    if negative_slope < 0:
        logger.warning(
            "Warning! 'slope' parameter should have positive value. "
            "Make sure this is intended!"
        )
    if activation == "selu":
        # See comment on Self-Normalizing Neural Networks
        # https://pytorch.org/docs/stable/generated/torch.nn.SELU.html#torch.nn.SELU
        activation = "linear"

    # Initialize basic layers
    if isinstance(
        module, nn.Linear | nn.Conv1d | nn.Conv2d | nn.Conv3d | nn.ConvTranspose2d
    ):
        # Weight initialization
        if weight_init == "xavier_uniform":
            nn.init.xavier_uniform_(module.weight)
        elif weight_init == "xavier_normal":
            nn.init.xavier_normal_(module.weight)
        elif weight_init == "kaiming_uniform":
            nn.init.kaiming_uniform_(module.weight, nonlinearity=activation)
        elif weight_init == "kaiming_normal":
            nn.init.kaiming_normal_(module.weight, nonlinearity=activation)
        elif weight_init == "orthogonal":
            nn.init.orthogonal_(module.weight)
        elif weight_init == "sparse":
            nn.init.sparse_(module.weight, sparsity=0.1)
        else:
            raise ValueError(
                f"Weight initialization '{weight_init}' not supported. "
                "Use 'xavier_uniform', 'xavier_normal', 'kaiming_uniform', "
                "'kaiming_normal', 'orthogonal' or 'sparse'."
            )

        # Bias initialization
        if hasattr(module, "bias") and module.bias is not None:
            if bias_init == "zeros":
                nn.init.zeros_(module.bias)
            elif bias_init == "ones":
                nn.init.ones_(module.bias)
            elif bias_init == "constant":
                nn.init.constant_(module.bias, 0.01)
            else:
                raise ValueError(
                    f"Bias initialization '{bias_init}' not supported. "
                    "Use 'zeros', 'ones' or 'constant' instead."
                )

    # Batch normalization initialization
    elif isinstance(module, nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d):
        if zero_batchnorm:
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # Layer normalization initialization
    elif isinstance(module, nn.LayerNorm | nn.GroupNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

    # Recurrent layer initialization
    elif isinstance(module, nn.LSTM | nn.GRU):
        for name, param in module.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    # Embedding layer initialization
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # Activation layer initialization
    elif isinstance(
        module, nn.ReLU | nn.LeakyReLU | nn.SELU | nn.Tanh | nn.Sigmoid | nn.Sequential
    ):
        pass

    else:
        logger.info(
            f"Module {module.__class__.__name__} not initialized "
            "due to missing init method. "
        )


def set_activation(
    network: nn.Sequential, activation: str, **kwargs: Any
) -> nn.Sequential:
    """
    Reset all activation functions in a sequential network
    to the specified activation function.

    Args:
        network: The network to modify.
        activation: The activation function to use.
        **kwargs: Additional arguments for the activation function.

    Returns:
        The modified network.

    Raises:
        ValueError: If the activation function is not supported.
    """
    negative_slope = kwargs.get("negative_slope", 0.0)

    for i, layer in enumerate(network):
        if isinstance(layer, nn.ReLU | nn.LeakyReLU | nn.SELU | nn.Tanh):
            if activation == "relu":
                network[i] = nn.ReLU()
            elif activation == "leaky_relu":
                network[i] = nn.LeakyReLU(negative_slope)
            elif activation == "selu":
                network[i] = nn.SELU()
            elif activation == "tanh":
                network[i] = nn.Tanh()
            else:
                raise ValueError(
                    f"Activation '{activation}' not supported. "
                    "Use 'relu', 'leaky_relu', 'selu' or 'tanh' instead."
                )

    return network
