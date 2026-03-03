"""Base PyTorch Lightning model for MCMLNet, inherited by all other models."""

import gc
from typing import Any

import lightning as pl
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig

from mcmlnet.training.data_loading.data_augmentation_classes import (
    apply_augmentations,
    instantiate_augmentations,
)
from mcmlnet.training.flop_and_gmac_counting import FLOPAndMACCounter
from mcmlnet.training.plotting import (
    display_parameter_marginals,
    display_spectra,
    plot_coeff_of_variation_vs_relative_error,
)
from mcmlnet.utils.logging import setup_logging
from mcmlnet.utils.network_init import (
    init_weights,
    matmul_precision,
    set_activation,
)

logger = setup_logging(level="info", logger_name=__name__)


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        mode: str,
        batch_size: int,
        max_augm_epochs: int,
        n_wavelengths: int,
        generator: DictConfig[str, Any],
        discriminator: DictConfig[str, Any],
        optimizer_gen: DictConfig[str, Any],
        scheduler_gen: DictConfig[str, Any],
        optimizer_dis: DictConfig[str, Any],
        scheduler_dis: DictConfig[str, Any],
        network_init: DictConfig[str, Any],
        loss_init: DictConfig[str, Any],
        data_module: pl.LightningDataModule | None,
    ) -> None:
        """
        Init for PyTorch network and datasets.

        Args:
            cfg: Hydra configuration for network and datasets
            mode: Mode of operation, e.g. train, tune, ...
            batch_size: Batch size for training
            max_augm_epochs: Number of epochs to augment data
            n_wavelengths: Number of wavelengths
            generator: Configuration for generator network
            discriminator: Configuration for discriminator network
            optimizer_gen: Configuration for generator optimizer
            scheduler_gen: Configuration for generator scheduler
            optimizer_dis: Configuration for discriminator optimizer
            scheduler_dis: Configuration for discriminator scheduler
            network_init: Configuration for network initialization
            loss_init: Configuration for loss initialization
            data_module: Data module for visual and compute logging
        """
        super().__init__()

        # Initialize parameters
        self.cfg = cfg
        self.mode = mode  # controls visual logging
        self.batch_size = batch_size
        self.max_augm_epochs = max_augm_epochs
        self.n_wavelengths = n_wavelengths

        self.optimizer_gen_kwargs = optimizer_gen
        self.scheduler_gen_kwargs = scheduler_gen
        self.optimizer_dis_kwargs = optimizer_dis
        self.scheduler_dis_kwargs = scheduler_dis
        self.loss_init = loss_init

        # Define matmul precision depending on trainer precision
        matmul_precision(self.cfg.trainer.precision)

        # Initialize networks
        self.generator = self._init_network(instantiate(generator), network_init)
        self.discriminator = (
            self._init_network(instantiate(discriminator), network_init)
            if discriminator
            else None
        )

        # Initialize dataloader for visual result logging
        self.plot_val_dataloader = data_module.val_dataloader() if data_module else None
        self.data_module = data_module

        # Initialize data augmentation (once)
        self.input_augments, self.output_augments = instantiate_augmentations(self.cfg)

        # Additional parameter and loss tracking
        self.test_step_outputs: list[torch.Tensor] = []
        self.predicted_vals: list[torch.Tensor] = []
        self.actual_vals: list[torch.Tensor] = []

        return None

    def _init_network(self, network: nn.Module, conf: DictConfig) -> nn.Module:
        """
        Initialize network with custom weight initialization and activation function.

        Args:
            network: Network to be initialized
            conf: Configuration for weight initialization and activation function

        Returns:
            Initialized network
        """
        # Initialize a random number generator
        rng = torch.Generator("cuda").manual_seed(42)

        # Define a function to initialize weights with a unique seed for each layer
        def init_weights_with_unique_seed(
            module: nn.Module, conf: DictConfig, rng: torch.Generator
        ) -> None:
            seed = int(
                torch.randint(0, 2**32 - 1, (1,), generator=rng, device="cuda").item()
            )
            init_weights(module=module, seed=seed, **conf)

        # Apply weight initialization and activation function
        network.apply(lambda module: init_weights_with_unique_seed(module, conf, rng))
        network.model = set_activation(network.model, **conf)
        if "64" in self.cfg.trainer.precision:
            network = network.double()
        return network

    def common_augment(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Wrapper around the augmentations function
        to use the same augmentations consistently.

        Args:
            input: Input tensor, e.g. optical parameters
            output: Output tensor, e.g. reflectance spectra

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Augmented input and output tensors
        """
        if self.current_epoch < self.max_augm_epochs:
            return apply_augmentations(
                input, output, self.input_augments, self.output_augments
            )
        else:
            return input, output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the generator network."""
        if "64" in self.cfg.trainer.precision:
            # use double precision for 64-bit training
            y = self.generator(x.double()).float()
        else:
            y = self.generator(x)
        return y

    def _related_work_losses(
        self, syn: torch.Tensor, output: torch.Tensor, name: str
    ) -> None:
        """Compute and log related work losses."""
        # compute relative (percent) error
        rel_error = torch.mean(torch.abs(syn - output) / output)
        self.log(f"rel_error_percent_{name}", rel_error * 100)
        # compute NMAE
        nmae = torch.mean(torch.abs(syn - output) / torch.mean(output))
        self.log(f"NMAE_{name}", nmae)

    def _simplified_calibration(self, syn: torch.Tensor, output: torch.Tensor) -> None:
        """Compute and log "calibration" metrics."""
        smaller = (syn < output).sum()
        larger = (syn > output).sum()
        self.log("calibration_ratio", smaller / (smaller + larger))

        # track model predictions and actual values for validation logging
        self.predicted_vals.append(syn)
        self.actual_vals.append(output)

    def gen_loss(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        compare: bool = False,
        step: str = "train",
    ) -> torch.Tensor:
        """
        Compute the generator loss.

        Args:
            input: Input tensor, e.g., optical parameters
            output: Output tensor, e.g., reflectance spectra
            compare: If True, loss will be made comparable
            step: Step name for logging (train, val, test)

        Returns:
            Computed loss for the batch
        """
        syn = self(input).squeeze()

        if compare:
            # transform back to normal scale if log intensity was used
            if self.cfg.preprocessing.log_intensity:
                syn, output = 10**syn, 10**output

            self._simplified_calibration(syn, output)
            self._related_work_losses(syn, output, name=step)

            return instantiate(self.loss_init.validate, output, syn)
        else:
            return instantiate(self.loss_init.train, output, syn)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step for the generator."""
        input, output = self.common_augment(*batch)
        gen_loss = self.gen_loss(input, output)
        self.log("gen_loss", gen_loss, prog_bar=True)
        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]["lr"])
        return gen_loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Validation step for the generator."""
        input, output = self.common_augment(*batch)
        val_loss = self.gen_loss(input, output, compare=True, step="val")
        self.log("val_loss", val_loss)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Test step for the generator."""
        input, output = self.common_augment(*batch)
        test_loss = self.gen_loss(input, output, compare=True, step="test")
        self.test_step_outputs.append(test_loss)

    def on_fit_start(self) -> None:
        """Initialize compute tracking and log the number of trainable parameters."""
        compute = FLOPAndMACCounter(self)
        n_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        gflops, _ = compute(
            self.data_module.train_dataloader(),  # type: ignore[union-attr]
            n_trainable_params,
            per_epoch=True,
        )
        # pick one method as reference FLOP counter
        self.flops_per_epoch = gflops["GFLOP FVCore"]

    def on_test_epoch_end(self) -> None:
        """Procedure for end of test epoch, logs average test loss."""
        outputs = self.test_step_outputs
        # compute and log the average loss
        avg_loss = torch.stack(outputs).mean()
        self.log("avg_test_loss", avg_loss.item())

        # free memory
        self.test_step_outputs.clear()

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizers and learning rate schedulers.

        Returns:
            A dictionary containing the optimizer and learning rate scheduler.
        """
        optimizer_gen = instantiate(
            self.optimizer_gen_kwargs, self.generator.parameters()
        )
        gen_scheduler = instantiate(self.scheduler_gen_kwargs, optimizer_gen)

        return {
            "optimizer": optimizer_gen,
            "lr_scheduler": gen_scheduler,
            "monitor": "gen_loss",
        }

    def on_train_epoch_end(self) -> None:
        pass

    def _validation_epoch_end_log(self) -> None:
        """Procedure for end of every epoch, logs validation metrics."""
        # Compute full validation set loss
        predicted_vals = torch.cat(self.predicted_vals, dim=0).squeeze()
        actual_vals = torch.cat(self.actual_vals, dim=0).squeeze()
        abs_diff = torch.abs(predicted_vals - actual_vals).flatten()
        self.log("val_loss_median", torch.median(abs_diff))
        self.log("val_loss_std", torch.std(abs_diff))
        # Compute val loss 95% PI and max
        self.log("val_loss_2.5", torch.quantile(abs_diff, 0.025))
        self.log("val_loss_97.5", torch.quantile(abs_diff, 0.975))
        self.log("val_loss_max", torch.max(abs_diff))

        # Repeat for the relative errors
        rel_error = abs_diff / actual_vals.flatten()
        self.log("val_rel_error", torch.mean(rel_error))
        self.log("val_rel_error_median", torch.median(rel_error))
        self.log("val_rel_error_std", torch.std(rel_error))
        self.log("val_rel_error_2.5", torch.quantile(rel_error, 0.025))
        self.log("val_rel_error_97.5", torch.quantile(rel_error, 0.975))
        self.log("val_rel_error_max", torch.max(rel_error))

        # Compute the calibration loss
        quantile_vals = torch.linspace(0, 1, 100).to(predicted_vals.device)
        predicted_quantiles = torch.quantile(predicted_vals, quantile_vals, dim=0)
        actual_quantiles = torch.quantile(actual_vals, quantile_vals, dim=0)
        calibration_loss = torch.mean(torch.abs(predicted_quantiles - actual_quantiles))
        self.log("calibration_loss", calibration_loss)

        # Clear the list for the next epoch
        self.predicted_vals = []
        self.actual_vals = []

        # Track compute
        self.log("PFLOP", self.flops_per_epoch / 1e6 * self.current_epoch)

    def on_validation_epoch_end(self) -> None:
        """
        Procedure for end of every epoch, visualizes five sample spectra.
        """
        self._validation_epoch_end_log()

        # Visual logging, skip for the hyperparameter tuning "tune" mode
        if self.mode == "train":
            # determine number of samples to use
            cutoff = 500 if self.n_wavelengths < 100 else 5

            # get samples from validation dataloader
            input, output = next(iter(self.plot_val_dataloader))  # type: ignore[arg-type]

            # handle different batch sizes
            if self.plot_val_dataloader.batch_size < cutoff:  # type: ignore[union-attr]
                # for small batches, collect multiple batches
                samples_collected = 0
                real_samples, gen_samples, inputs = [], [], []

                for batch_input, batch_output in self.plot_val_dataloader:  # type: ignore[union-attr]
                    batch_input, batch_output = self.common_augment(
                        batch_input, batch_output
                    )

                    with torch.no_grad():
                        batch_gen = self(batch_input.to(self.device))

                    inputs.append(batch_input)
                    real_samples.append(batch_output)
                    gen_samples.append(batch_gen)

                    samples_collected += self.batch_size
                    if samples_collected >= cutoff:
                        break

                input = torch.cat(inputs, dim=0)
                output = torch.cat(real_samples, dim=0)
                gen = torch.cat(gen_samples, dim=0)
            else:
                # for large batches, use first :cutoff samples
                input, output = self.common_augment(input, output)
                with torch.no_grad():
                    gen = self(input.to(self.device))

            input = input.squeeze()[:cutoff]
            output = output.squeeze()[:cutoff]
            gen = gen.squeeze()[:cutoff]

            # change order of parameters to layer-wise ordering (expects three layers)
            n_params_yaml = self.cfg.n_params
            input = torch.cat(
                ([input[:, i::n_params_yaml] for i in range(n_params_yaml)]), dim=1
            ).view(-1, input.shape[-1])

            # visual logging of input marginals
            if input.ndim == 3:
                input = input.view(-1, input.shape[-1])

            self.logger.experiment.add_figure(
                "input_marginals",
                display_parameter_marginals(
                    input,
                    None,
                ),
                self.current_epoch,
            )

            # visual logging of output spectra
            if self.cfg.preprocessing.log_intensity:
                self.logger.experiment.add_figure(
                    "log_intensity",
                    display_spectra(
                        output[:5],
                        gen[:5],
                        n_rows=1,
                        n_cols=5,
                    ),
                    self.current_epoch,
                )
                # convert back to normal scale for coefficient of variation plot
                gen, output = 10**gen, 10**output

            self.logger.experiment.add_figure(
                "coeff_var_plot",
                plot_coeff_of_variation_vs_relative_error(gen, output),
                self.current_epoch,
            )

            # display five images each in grid and log grid
            if output.ndim == 1 or input.ndim == 1:
                # manual reshape to spectra-like shape using factorization,
                # stitching together multiple spectra for sample diversity
                rows, cols = self._find_reshape_factors(
                    output.shape[0], min_first_dim=5
                )
                output = output.view(rows, cols)
                gen = gen.view(rows, cols)

            self.logger.experiment.add_figure(
                "synthesized_spectra",  # or optical parameters
                display_spectra(
                    output[:5],
                    gen[:5],
                    n_rows=1,
                    n_cols=5,
                ),
                self.current_epoch,
            )

            # as mentioned in https://github.com/pytorch/pytorch/issues/67978#issuecomment-1661986812
            gc.collect()

        else:
            pass

    def _find_reshape_factors(
        self, batch_length: int, min_first_dim: int = 5
    ) -> tuple[int, int]:
        """
        Find two factors of batch_length where the first factor is >= min_first_dim.

        Args:
            batch_length: The total batch length to factorize
            min_first_dim: Minimum value for the first dimension (default: 5)

        Returns:
            Tuple of (rows, cols) where rows >= min_first_dim
                and rows * cols = batch_length
        """
        # Find all factors of batch_length
        factors = []
        for i in range(1, int(batch_length**0.5) + 1):
            if batch_length % i == 0:
                factors.extend([i, batch_length // i])

        # Remove duplicates and sort
        factors = sorted(set(factors))

        # Find the best pair where first factor >= min_first_dim
        for factor in factors:
            if factor >= min_first_dim:
                complement = batch_length // factor
                return factor, complement

        # If no factor >= min_first_dim found, return the closest approximation
        # Find the largest factor < min_first_dim and use min_first_dim if possible
        if batch_length % min_first_dim == 0:
            return min_first_dim, batch_length // min_first_dim

        # Fallback: use the largest factor and its complement
        return factors[-1], batch_length // factors[-1]
