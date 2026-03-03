"""Test network constructor and BaseModel and KANModel integration."""

import copy
from collections.abc import Generator
from itertools import product

import pytest
import torch
from omegaconf import OmegaConf

from mcmlnet.training.models.base_model import BaseModel
from mcmlnet.training.models.kan_model import KANModel
from mcmlnet.training.models.network_constructor import ForwardSurrogateModel


class DummyDataModule:
    """Minimal dummy data module for integration tests."""

    def train_dataloader(
        self,
    ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        """Return a simple iterable of (input, output) batches."""
        for _ in range(2):
            yield torch.randn(4, 15), torch.randn(4, 100)

    def val_dataloader(
        self,
    ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        """Return a simple iterable of (input, output) batches."""
        for _ in range(2):
            yield torch.randn(4, 15), torch.randn(4, 100)


@pytest.mark.parametrize(  # type: ignore[misc]
    "arch_type,sigmoid,spectra",
    list(product(["simple", "normalized", "minimal"], [True, False], [True, False])),
)
def test_forward_surrogate_model_valid(
    arch_type: str, sigmoid: bool, spectra: bool
) -> None:
    """Test valid configurations of ForwardSurrogateModel."""
    model = ForwardSurrogateModel(
        layers=[32, 100, 1],
        n_params=15,
        arch_type=arch_type,
        p_dropout=0.1,
        sigmoid=sigmoid,
        spectra=spectra,
    )
    if sigmoid:
        assert isinstance(model.model[-1], torch.nn.Sigmoid)
    x = torch.randn(2, 100, 15)
    out = model(x)
    if spectra:
        assert out.shape == (2, 100)
    else:
        assert out.shape == (200, 1)
    info = model.get_model_info()
    assert "total_parameters" in info
    assert "trainable_parameters" in info
    assert "input_dim" in info
    assert "spectra_mode" in info
    assert "architecture" in info


def test_forward_surrogate_model_invalid() -> None:
    """Test invalid configurations of ForwardSurrogateModel."""
    with pytest.raises(ValueError, match="Layers list cannot be empty"):
        ForwardSurrogateModel(layers=[], n_params=15)
    with pytest.raises(ValueError, match="Number of parameters must be positive"):
        ForwardSurrogateModel(layers=[15, 100], n_params=0)
    with pytest.raises(ValueError, match="Dropout probability must be in"):
        ForwardSurrogateModel(layers=[15, 100], n_params=15, p_dropout=1.5)
    with pytest.raises(ValueError, match="Dropout probability must be in"):
        ForwardSurrogateModel(layers=[15, 100], n_params=15, p_dropout=-0.1)
    with pytest.raises(ValueError, match="All layer sizes must be positive"):
        ForwardSurrogateModel(layers=[0, 100], n_params=15)
    with pytest.raises(ValueError, match="All layer sizes must be positive"):
        ForwardSurrogateModel(layers=[-1, 100], n_params=15)
    with pytest.raises(
        AssertionError, match="Last network layer must have dimension in "
    ):
        ForwardSurrogateModel(layers=[32, 99], n_params=15)
    with pytest.raises(
        NotImplementedError, match="'arch_type' unknown not implemented"
    ):
        ForwardSurrogateModel(layers=[32, 100], n_params=15, arch_type="unknown")


class TestPLModels:
    """Test PyTorch Lightning models."""

    def setup_method(self) -> None:
        """Setup common test variables."""
        self.generator = {
            "_target_": "mcmlnet.training.models.network_constructor.ForwardSurrogateModel",  # noqa: E501
            "layers": [32, 100],
            "n_params": 15,
        }
        # Generic config object
        self.cfg = OmegaConf.create(
            {
                "trainer": {
                    "accelerator": "cpu",
                    "precision": "32-true",
                    "max_epochs": 2,
                    "log_every_n_steps": 1,
                },
                "name": "test",
                "n_params": 15,
                "input_augmentations": {},
                "output_augmentations": {},
            }
        )
        # Loss initialization
        self.loss_init = OmegaConf.create(
            {
                "train": {"_target_": "torch.nn.functional.l1_loss"},
                "validate": {"_target_": "torch.nn.functional.mse_loss"},
                "regularization_weight": None,
            }
        )
        # Other DictConfig-like objects
        self.optimizer = {"_target_": "torch.optim.Adam"}
        self.scheduler = {"_target_": "torch.optim.lr_scheduler.StepLR", "step_size": 1}
        self.network_init = {
            "activation": "relu",
            "weight_init": "xavier_uniform",
            "bias_init": "zeros",
        }
        # Dummy data module
        self.data_module = DummyDataModule()

    def test_base_model_integration(self) -> None:
        """Simple BaseModel integration test."""
        model = BaseModel(
            cfg=self.cfg,
            mode="train",
            batch_size=4,
            max_augm_epochs=2,
            n_wavelengths=100,
            generator=self.generator,
            discriminator=None,
            optimizer_gen=self.optimizer,
            scheduler_gen=self.scheduler,
            optimizer_dis=None,
            scheduler_dis=None,
            network_init=self.network_init,
            loss_init=self.loss_init,
            data_module=self.data_module,
        )

        # Run basic checks (possible without PyTorch Lightning Trainer)
        x = torch.randn(2, 15)
        y = model(x)
        assert y.shape == (2, 100)
        batch = (torch.randn(4, 15), torch.randn(4, 100))
        out = model.gen_loss(batch[0], batch[1])
        assert isinstance(out, torch.Tensor)
        assert out.shape == torch.Size([])

        # Check optimizer configuration
        optim = model.configure_optimizers()
        assert "optimizer" in optim
        assert "lr_scheduler" in optim
        rows, cols = model._find_reshape_factors(20)
        assert rows * cols == 20

    def test_kan_model_integration(self) -> None:
        """Simple KANModel integration test."""
        # Update generator to use KAN
        generator = copy.deepcopy(self.generator)
        generator["_target_"] = "mcmlnet.training.models.kan.KAN"
        generator["layers_hidden"] = [15, 100]
        del generator["layers"]
        del generator["n_params"]

        model = KANModel(
            cfg=self.cfg,
            mode="train",
            batch_size=4,
            max_augm_epochs=2,
            n_wavelengths=100,
            generator=generator,
            discriminator=None,
            optimizer_gen=self.optimizer,
            scheduler_gen=self.scheduler,
            optimizer_dis=None,
            scheduler_dis=None,
            network_init=self.network_init,
            loss_init=self.loss_init,
            data_module=self.data_module,
        )

        # Run basic checks (possible without PyTorch Lightning Trainer)
        x = torch.randn(2, 15)
        y = model(x)
        assert y.shape == (2, 100)
        batch = (torch.randn(4, 15), torch.randn(4, 100))
        out = model.gen_loss(batch[0], batch[1])
        assert isinstance(out, torch.Tensor)
        assert out.shape == torch.Size([])

        # Check optimizer configuration
        optim = model.configure_optimizers()
        assert "optimizer" in optim
        assert "lr_scheduler" in optim
        rows, cols = model._find_reshape_factors(20)
        assert rows * cols == 20
