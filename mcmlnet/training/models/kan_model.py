"""KAN PyTorch Lightning model, inherited from the BaseModel class."""

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig

from mcmlnet.training.models.base_model import BaseModel
from mcmlnet.training.plotting import (
    display_parameter_marginals,
    display_spectra,
    plot_coeff_of_variation_vs_relative_error,
)


class KANModel(BaseModel):
    def _init_network(self, network: nn.Module, conf: DictConfig) -> nn.Module:
        """
        Initialize network with custom weight initialization and activation function.

        Args:
            network: Network to be initialized
            conf: Configuration for weight initialization and activation function

        Returns:
            Initialized network
        """
        if "64" in self.cfg.trainer.precision:
            network = network.double()
        return network

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
            # Undo log intensity scaling for reflectance spectra in comparison
            if self.cfg.preprocessing.log_intensity:
                syn, output = 10**syn, 10**output

            self._simplified_calibration(syn, output)
            self._related_work_losses(syn, output, name=step)

            return instantiate(self.loss_init.validate, output, syn)
        else:
            # Instantiate the losses
            regular_loss = instantiate(self.loss_init.train, output, syn)

            if self.loss_init.regularization_weight is not None:
                kan_regularization_loss = self.generator.regularization_loss()
                self.log("kan_regularization_loss", kan_regularization_loss)
                regular_loss += (
                    self.loss_init.regularization_weight * kan_regularization_loss
                )
            return regular_loss

    def on_train_epoch_end(self) -> None:
        pass

    def on_validation_epoch_end(self) -> None:
        """
        Procedure for end of every epoch, visualizes five sample spectra.

        """
        self._validation_epoch_end_log()

        # Visual logging, skip for the hyperparameter tuning "tune" mode
        if self.mode == "train":
            # determine number of samples to use
            cutoff = 100 if self.n_wavelengths < 100 else 5

            # get samples from validation dataloader
            input, output = next(iter(self.plot_val_dataloader))

            # handle different batch sizes
            if self.batch_size < cutoff:
                # for small batches, collect multiple batches
                samples_collected = 0
                real_samples, gen_samples, inputs = [], [], []

                for batch_input, batch_output in self.plot_val_dataloader:
                    # extract single samples if batch_size > 1
                    if self.batch_size > 1:
                        batch_input, batch_output = batch_input[0], batch_output[0]

                    batch_input, batch_output = (
                        batch_input.unsqueeze(0),
                        batch_output.unsqueeze(0),
                    )
                    batch_input, batch_output = self.common_augment(
                        batch_input, batch_output
                    )

                    with torch.no_grad():
                        batch_gen = self.generator(batch_input.to(self.device))

                    inputs.append(batch_input[0])
                    real_samples.append(batch_output[0])
                    gen_samples.append(batch_gen[0])

                    samples_collected += 1
                    if samples_collected >= cutoff:
                        break

                input = torch.stack(inputs, dim=0)
                output = torch.stack(real_samples, dim=0)
                gen = torch.stack(gen_samples, dim=0)
            else:
                # for large batches, use first :cutoff samples
                input, output = self.common_augment(
                    input[:cutoff], output[:cutoff].unsqueeze(1)
                )
                with torch.no_grad():
                    gen = self.generator(input.to(self.device))

            # change order of parameters to layer-wise ordering (expects three layers)
            n_params_yaml = self.cfg.n_params
            input = torch.cat(
                ([input[:, i::n_params_yaml] for i in range(n_params_yaml)]), dim=1
            ).view(-1, input.shape[-1])

            if self.cfg.preprocessing.log_intensity:
                # plot unprocessed log intensity spectra
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

            # visual logging of input marginals
            self.logger.experiment.add_figure(
                "input_marginals",
                display_parameter_marginals(
                    input,
                    None,
                ),
                self.current_epoch,
            )

        else:
            pass
