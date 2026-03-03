"""Network constructor for fully connected surrogate models in PyTorch Lightning."""

from typing import ClassVar

import lightning as pl
import torch
import torch.nn as nn

from mcmlnet.utils.logging import setup_logging

logger = setup_logging(level="info", logger_name=__name__)


def fc_building_blocks(
    arch_type: str,
    in_dim: int,
    out_dim: int,
    p_dropout: float = 0.05,
    negative_slope: float = 0.2,
) -> list[nn.Module]:
    """Create building blocks for fully connected layers.

    Args:
        arch_type: Type of architecture ('simple', 'normalized', 'minimal')
        in_dim: Input dimension
        out_dim: Output dimension
        p_dropout: Dropout probability (default: 0.05)
        negative_slope: Negative slope for LeakyReLU (default: 0.2)

    Returns:
        List of PyTorch modules forming the building block

    Raises:
        ValueError: If dimensions are invalid
        NotImplementedError: If arch_type is not supported
    """
    # Validate inputs
    if in_dim <= 0 or out_dim <= 0:
        raise ValueError(
            f"Dimensions must be positive, got in_dim={in_dim}, out_dim={out_dim}"
        )
    if not 0 <= p_dropout <= 1:
        raise ValueError(f"Dropout probability must be in [0, 1], got {p_dropout}")
    if negative_slope < 0:
        raise ValueError(
            f"LeakyReLU negative slope must be non-negative, got {negative_slope}"
        )

    arch_type = arch_type.lower()

    if arch_type == "simple":
        block = [
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(p=p_dropout),
        ]
    elif arch_type == "normalized":
        # goal: prevent nan's for deep networks
        block = [
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(p=p_dropout),
            nn.BatchNorm1d(out_dim),
        ]
    elif arch_type == "minimal":
        if p_dropout != 0:
            logger.warning(
                "Minimal architecture does not support dropout. Setting to 0."
            )
        # add only linear layer without activation
        # if output dimension is 1 (final layer)
        if out_dim == 1:
            block = [nn.Linear(in_dim, out_dim)]
        else:
            block = [
                nn.Linear(in_dim, out_dim),
                nn.LeakyReLU(negative_slope),
            ]
    else:
        available_types = ["simple", "normalized", "minimal"]
        raise NotImplementedError(
            f"'arch_type' {arch_type} not implemented. "
            f"Available types: {available_types}"
        )

    return block


class ForwardSurrogateModel(pl.LightningModule):
    """Fully connected network for predicting optical spectra from physical properties.

    This model takes physical parameters as input and predicts optical spectra
    or single wavelength values as output.
    """

    # Valid output dimensions for different use cases
    VALID_OUTPUT_DIMS: ClassVar[set[int]] = {1, 15, 100, 351}

    def __init__(
        self,
        layers: list[int],
        n_params: int = 15,
        arch_type: str = "simple",
        p_dropout: float = 0.05,
        sigmoid: bool = True,
        spectra: bool = True,
        negative_slope: float = 0.2,
    ) -> None:
        """Initialize the fully connected surrogate model.

        Args:
            layers: List of layer sizes (excluding input layer)
            n_params: Dimensionality of physical properties (input size)
            arch_type: Type of basic building block architecture
            p_dropout: Dropout probability
            sigmoid: Whether to apply sigmoid activation at output
            spectra: Whether to use spectra as batch input
            negative_slope: Negative slope for LeakyReLU activation

        Raises:
            ValueError: If layer configuration is invalid
            AssertionError: If output dimension is not supported
        """
        super().__init__()

        # Validate inputs
        if not layers:
            raise ValueError("Layers list cannot be empty")
        if n_params <= 0:
            raise ValueError(f"Number of parameters must be positive, got {n_params}")
        if not 0 <= p_dropout <= 1:
            raise ValueError(f"Dropout probability must be in [0, 1], got {p_dropout}")
        if any(layer <= 0 for layer in layers):
            raise ValueError("All layer sizes must be positive")

        # Validate output dimension
        output_dim = layers[-1]
        if output_dim not in self.VALID_OUTPUT_DIMS:
            raise AssertionError(
                f"Last network layer must have dimension in {self.VALID_OUTPUT_DIMS}, "
                f"but found {output_dim}!"
            )

        # Store configuration
        self.n_parameters = n_params
        self.spectra = spectra

        # Define MLP structure
        layers.insert(0, self.n_parameters)
        model = []
        for i in range(len(layers) - 1):
            model.append(
                fc_building_blocks(
                    arch_type, layers[i], layers[i + 1], p_dropout, negative_slope
                )
            )

        # Flatten and convert to Sequential
        self.model = nn.Sequential(*[item for sublist in model for item in sublist])
        if sigmoid:
            self.model.add_module("Sigmoid", nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, n_parameters) or
               (batch_size, n_wavelengths, n_parameters) if spectra=True

        Returns:
            Output reflectance tensor
        """
        if self.spectra:
            return self.model(x.view(-1, self.n_parameters)).view(len(x), -1)
        else:
            return self.model(x.view(-1, self.n_parameters))

    def get_model_info(self) -> dict:
        """Get information about the model architecture.

        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "input_dim": self.n_parameters,
            "spectra_mode": self.spectra,
            "architecture": str(self.model),
        }
