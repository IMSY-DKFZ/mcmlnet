"""Convenience functions for model loading and prediction."""

import os
import warnings

import lightning as pl
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich.progress import track
from torch.utils.data import DataLoader, TensorDataset

from mcmlnet.training.data_loading.preprocessing import PreProcessor
from mcmlnet.training.models.base_model import BaseModel
from mcmlnet.utils.load_configs import load_config
from mcmlnet.utils.logging import setup_logging

logger = setup_logging(level="info", logger_name=__name__)


PARAM_BOUNDS = {
    "mu_a": (0.2, 8e4),  # [m^-1]
    "mu_s": (400.0, 4.5e5),  # [m^-1]
    "g": (0.8, 0.95),
    "n": (1.33, 1.54),
    "d": (2e-5, 2e-3),  # [m]
}


def _warn_out_of_range(stacked: np.ndarray) -> None:
    # Reshape stacked data: (batch_size, 15) -> (batch_size, 5 params, 3 layers)
    grouped = stacked.reshape(stacked.shape[0], 5, 3)
    names = ["mu_a", "mu_s", "g", "n", "d"]

    for i, name in enumerate(names):
        vals = grouped[:, i, :]
        lo, hi = PARAM_BOUNDS[name]

        out_of_range = (~np.isfinite(vals)) | (vals < lo) | (vals > hi)

        if out_of_range.any():
            clipped = vals[out_of_range]
            warnings.warn(
                f"{name} outside [{lo}, {hi}] for {out_of_range.sum()} samples "
                f"(min={clipped.min():.3g}, max={clipped.max():.3g})",
                stacklevel=2,
            )


def load_trained_model(
    base_path: str, checkpoint_path: str, model_type: type[BaseModel]
) -> tuple[BaseModel, PreProcessor, DictConfig]:
    """Load a trained model with its configuration and preprocessor.

    Args:
        base_path: Path to the model directory.
        checkpoint_path: Path to the checkpoint file.
        model_type: The model class to load.

    Returns:
        Tuple of (loaded model, preprocessor, and config).
    """
    # Load config
    loaded_config = OmegaConf.load(os.path.join(base_path, "config_log.yaml"))

    # Load preprocessor and data module
    preprocessor = instantiate(loaded_config.preprocessing)
    # Turn saved norms into tensors
    if preprocessor.norm_1 is not None:
        preprocessor.norm_1 = torch.tensor(preprocessor.norm_1)
    if preprocessor.norm_2 is not None:
        preprocessor.norm_2 = torch.tensor(preprocessor.norm_2)
    logger.info(f"Preprocessor norm.s: {preprocessor.norm_1}, {preprocessor.norm_2}")

    # Load model with config
    model = model_type.load_from_checkpoint(
        os.path.join(base_path, checkpoint_path),
        **loaded_config.model,
        cfg=loaded_config,
        data_module=None,
    )
    # Eval and freeze model
    model.eval()
    model.freeze()

    return model, preprocessor, loaded_config


def prepare_surrogate_model_data(
    preprocessor: PreProcessor,
    cfg: DictConfig,
    data: torch.Tensor,
) -> torch.Tensor:
    """Prepare data for surrogate model prediction.

    Args:
        preprocessor: The preprocessor to use.
        cfg: The model configuration.
        data: The input data.

    Returns:
        Prepared data for prediction.
    """
    data = preprocessor(data)

    if cfg.dataset.thick_deepest_layer:
        # Set the thickness of the deepest layer to 0
        data[..., -2] = 0

    return data


def predict_in_batches(
    model: pl.LightningModule,
    data: torch.Tensor,
    batch_size: int = -1,
    progress_bar: bool = True,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Make predictions in batches to avoid memory issues.

    Args:
        model: The model to use for prediction.
        data: The PREPROCESSED/ TRANSFORMED input data.
        batch_size: Batch size for prediction (-1 for single batch).
        progress_bar: Whether to use rich.progress track or not.
        requires_grad: Whether to retain the gradient or not.

    Returns:
        Model predictions.
    """
    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(data)

    if batch_size == -1:
        batch_size = len(data)

    predictions = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    ctx = torch.no_grad() if not requires_grad else torch.enable_grad()
    with ctx:
        # Reduce batch size while out of memory error occurs
        while batch_size > 1:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            if progress_bar:
                dataloader = track(
                    dataloader, description="Making predictions", transient=True
                )
            try:
                for _i, batch in enumerate(dataloader):
                    # Get first element from batch-tuple
                    preds = model(batch[0].to(device))
                    predictions.append(preds if requires_grad else preds.detach().cpu())
            except torch.OutOfMemoryError:
                batch_size //= 2
                logger.warning(
                    "torch out of memory! Reducing batch size to "
                    f"{batch_size} and repeating ..."
                )
            else:
                break

    # Concatenate all predictions
    return torch.cat(predictions, dim=0)


def batch_inputs_to_three_layer_model(
    mu_a: np.ndarray,
    mu_s: np.ndarray,
    g: np.ndarray,
    n: np.ndarray,
    d: np.ndarray | None = None,
    dummy_thickness: float = 0.001,
) -> np.ndarray:
    """
    Build three-layer model inputs from 1- or 2-layer params.

    Args:
        mu_a, mu_s, g, n: shape (B,) or (B, L) with L in {1, 2, 3}.
        d: thickness; if None a dummy thickness is used. Otherwise,
            the shape must be the same as for the other provided parameters.
            NOTE: The lowest layer is ALWAYS assumed to be semi-infinite and was
            modelled to be 20 cm in Monte Carlo simulations during training-time.
            Therefore, in `prepare_surrogate_model_data`, d_3 is always set to zero.
        dummy_thickness: default thickness for all layers if not specified otherwise.

    Returns:
        np.ndarray with shape (B, 15): [3x mu_a, 3x mu_s, 3x g, 3x n, 3x d]
    """

    def ensure_2d(x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return x[:, None]
        if x.ndim != 2:
            raise ValueError("Inputs must be 1D or 2D!")
        return x

    # Collect parameters and assert correctness of shapes
    mu_a, mu_s, g, n = map(ensure_2d, (mu_a, mu_s, g, n))
    _batch, num_layers = mu_a.shape

    if not (mu_a.shape == mu_s.shape == g.shape == n.shape):
        raise ValueError("mu_a, mu_s, g, n must share the same shape!")
    if num_layers not in (1, 2, 3):
        raise ValueError("Only 1, 2, or 3 layers are supported!")

    # Define the tissue layer structure (thickness)
    if d is None:
        d = np.full_like(mu_a, dummy_thickness)
    else:
        if not d.shape == mu_a.shape:
            raise ValueError("d must share the same shape with the other parameters!")
    d = ensure_2d(d)

    # Pad the parameters to match our expected three-layer structure
    # by repeating the deepest specified layer.
    def pad_params(x: np.ndarray, fill_last: float | None = None) -> np.ndarray:
        if num_layers == 3:
            return x
        if num_layers == 2:
            # Repeat last layer if no fill value is provided
            last = (
                np.full_like(x[:, :1], fill_last)
                if fill_last is not None
                else x[:, -1:]
            )
            return np.concatenate([x, last], axis=1)

        return np.repeat(x, 3, axis=1)

    mu_a = pad_params(mu_a)
    mu_s = pad_params(mu_s)
    g = pad_params(g)
    n = pad_params(n)
    d = pad_params(d, fill_last=dummy_thickness)

    # Concatenate the values into the expected physical order [3x mu_a, 3x mu_s, ...]
    data = np.concatenate([mu_a, mu_s, g, n, d], axis=1)

    return data


def run_model_from_physical_data(
    mu_a: np.ndarray,
    mu_s: np.ndarray,
    g: np.ndarray,
    n: np.ndarray,
    d: np.ndarray | None,
    add_specular_reflectance: bool = False,
    model_tuple: tuple[BaseModel, PreProcessor, DictConfig] | None = None,
    model_base_path: str | None = None,
    model_checkpoint_path: str | None = None,
    inference_batch_size: int = 1000,
) -> np.ndarray:
    """
    Run the model on the given data. The data is preprocessed and postprocessed
    according to the model's requirements. Does NOT support autograd graph.

    Args:
        mu_a (np.ndarray): The absorption coefficient in 1/m as a 1D or 2D or 2D array.
        mu_s (np.ndarray): The scattering coefficient in 1/m as a 1D array.
        g (np.ndarray): The anisotropy as a 1D or 2D array.
        n (np.ndarray): The refractive index as a 1D or 2D array.
        d (np.ndarray | None): Thickness as 1D or 2D array.
            NOTE: thickness d is not always required, as the model
            uses a "thick" deepest layer of 20 cm
        add_specular_reflectance: Whether to add air as the first layer.
        model_tuple (tuple[BaseModel, PreProcessor, DictConfig]): Model inference tuple.
        model_base_path (str): Absolute path to the model's base folder.
        model_checkpoint_path (str): Relative path to the model's checkpoint.
        inference_batch_size (int): Batch size for model inference.
    Returns:
        np.ndarray: The model's prediction as a 1D array.
    """
    # Load default configuration
    if model_base_path is None or model_checkpoint_path is None:
        cfg = load_config()
        model_base_path = os.path.join(
            os.environ["data_dir"], cfg["surrogate"]["issi_model"]["base_path"]
        )
        model_checkpoint_path = cfg["surrogate"]["issi_model"]["checkpoint_path"]

    # Load trained model
    if model_tuple is not None:
        # Pass loaded model to allow front-loading the loading effort
        model, preprocessor, model_cfg = model_tuple
    else:
        model, preprocessor, model_cfg = load_trained_model(
            model_base_path, model_checkpoint_path, BaseModel
        )

    # Create default three-layer tissue model
    data = batch_inputs_to_three_layer_model(mu_a, mu_s, g, n, d)
    _warn_out_of_range(data)
    # Add wavelength axis and dummy outputs for shape compatibility and preprocess data
    data = torch.from_numpy(data).unsqueeze(1).float()
    data = torch.cat((data, torch.zeros_like(data)[..., [0]]), dim=-1)
    data = prepare_surrogate_model_data(preprocessor, model_cfg, data)[..., :-1]

    reflectance = predict_in_batches(
        model,
        data,
        batch_size=inference_batch_size,
        requires_grad=False,
    )

    if add_specular_reflectance:
        n_air = 1.0
        # Collect top tissue layer refractive index
        n_surface = torch.as_tensor(
            n, device=reflectance.device, dtype=reflectance.dtype
        )
        if n_surface.ndim == 1:
            n_surface = n_surface[:, None]
        n_surface = n_surface[:, :1]

        specular = ((n_surface - n_air) ** 2) / ((n_surface + n_air) ** 2)
        reflectance += specular.expand_as(reflectance)

    return reflectance.detach().cpu().numpy()
