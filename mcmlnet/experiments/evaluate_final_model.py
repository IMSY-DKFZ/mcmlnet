import os

import lightning as pl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from omegaconf import DictConfig
from scipy.stats import sem

from mcmlnet.experiments.plotting import setup_plot_style
from mcmlnet.experiments.utils import compute_bootstrap_ci, compute_reordering_array
from mcmlnet.training.data_loading.preprocessing import PreProcessor
from mcmlnet.training.models.base_model import BaseModel
from mcmlnet.utils.convenience import (
    load_trained_model,
    predict_in_batches,
    prepare_surrogate_model_data,
)
from mcmlnet.utils.load_configs import load_config
from mcmlnet.utils.loading import SimulationDataLoader
from mcmlnet.utils.logging import setup_logging
from mcmlnet.utils.metrics import (
    custom_correlation_coefficient_diagonal,
    custom_r2_score,
)
from mcmlnet.utils.process_spectra import coeff_of_variation

setup_plot_style()
logger = setup_logging(level="info", logger_name=__name__)


if not os.path.exists(os.environ["plot_save_dir"]):
    os.makedirs(os.environ["plot_save_dir"])


def compute_r2_old(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the R^2 score."""
    sqr = np.sum((y_true - y_pred) ** 2)
    sqt = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - sqr / sqt)


def compute_adjusted_r2(
    y_true: torch.Tensor, y_pred: torch.Tensor, model: pl.LightningModule
) -> float:
    """Compute the adjusted R^2 score."""
    n_param_model = sum(p.numel() for p in model.parameters())
    n_obs = len(y_true)
    return float(
        1
        - (1 - custom_r2_score(y_true, y_pred).mean().item())
        * (n_obs - 1)
        / (n_obs - n_param_model - 1)
    )


def rel_error_PI(
    p: np.ndarray,
    n_photons: int,
    lower: float = 0.025,
    upper: float = 0.975,
    n_samples: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate relative error prediction interval."""
    torch.manual_seed(42)
    binom = torch.distributions.binomial.Binomial(
        total_count=n_photons, probs=torch.tensor(p, dtype=torch.float32)
    )
    samples = binom.sample((n_samples,)).numpy()
    q_lower = np.quantile(np.abs(samples / n_photons - p) / p, lower, axis=0)
    q_upper = np.quantile(np.abs(samples / n_photons - p) / p, upper, axis=0)

    return 100 * q_lower, 100 * q_upper


def load_and_process_data(
    preprocessor: PreProcessor, cfg: DictConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load and process all datasets."""
    # Load physical and physiological data
    physical_data = SimulationDataLoader().load_physical_simulation_data(
        dataset_name="raw/base_physio_and_physical_simulations/physical_generalization_100M_photons.parquet",
        n_wavelengths=351,
    )
    physiological_data = SimulationDataLoader().load_simulation_data(
        dataset_name="raw/base_physio_and_physical_simulations/physiological_training_100M_photons_1M_samples.parquet",
    )

    # Get test splits
    physical_data = physical_data[
        preprocessor.consistent_data_split_ids(physical_data, mode="test")
    ]
    physiological_data = physiological_data[
        preprocessor.consistent_data_split_ids(physiological_data, mode="test")
    ]

    # Manually load 1 billion photon dataset
    batch_folder = os.path.join(
        os.environ["data_dir"],
        "raw/base_physio_and_physical_simulations/physiological_training_100M_1M_samples_test_10x_photons.parquet",
    )
    physiological_1000M_df = pd.read_parquet(batch_folder)
    physiological_1000M_data = torch.from_numpy(
        SimulationDataLoader().simulation_to_standard_numpy(physiological_1000M_df)
    )

    # Manually load 10M photon dataset to correctly order the data
    batch_folder = os.path.join(
        os.environ["data_dir"],
        "raw/base_physio_and_physical_simulations/physiological_ablation_10M_photons.parquet",
    )
    physiological_10M_df = pd.read_parquet(batch_folder)
    physiological_10M_data = torch.from_numpy(
        SimulationDataLoader().simulation_to_standard_numpy(physiological_10M_df)
    )

    # Load or compute reordering array
    cache_path = os.path.join(
        os.environ["cache_dir"], "0_1-10M_physiological_sims_data_reordering_array.txt"
    )
    try:
        matched_indices = np.loadtxt(cache_path, dtype=int)
    except FileNotFoundError:
        matched_indices = compute_reordering_array(
            physiological_1000M_data, physiological_10M_data, cache_path
        )

    # Process data for surrogate model, with re-sorted physiological data
    physical_data = prepare_surrogate_model_data(preprocessor, cfg, physical_data)
    physiological_data = prepare_surrogate_model_data(
        preprocessor, cfg, physiological_data
    )
    physiological_10M_data = prepare_surrogate_model_data(
        preprocessor, cfg, physiological_10M_data[matched_indices]
    )
    physiological_1000M_data = prepare_surrogate_model_data(
        preprocessor, cfg, physiological_1000M_data
    )

    return (
        physical_data,
        physiological_data,
        physiological_1000M_data,
        physiological_10M_data,
    )


def evaluate_model(
    model: pl.LightningModule, datasets: list[tuple[torch.Tensor, str]]
) -> list[tuple[torch.Tensor, torch.Tensor, str]]:
    """Evaluate model on multiple datasets and return predictions."""
    results = []
    for data, name in datasets:
        y_pred = predict_in_batches(
            model, data[..., :-1], batch_size=1000 if "physiological" in name else 100
        )
        y_true = data[..., -1].reshape(-1, y_pred.shape[-1])
        results.append((y_true, y_pred, name))
        logger.info(f"Size of {name}: {data.shape}")
        logger.info(
            f"Min/Max reflectance values: {y_true.min().item():.5f}, "
            f"{y_true.max().item():.5f}"
        )
        logger.info(
            f"Min/Max predicted values: {y_pred.min().item():.5f}, "
            f"{y_pred.max().item():.5f}"
        )
    return results


def compute_metrics(
    results: list[tuple[torch.Tensor, torch.Tensor, str]], model: pl.LightningModule
) -> None:
    """Compute MAE and R² metrics for all results."""
    labels = [
        "Physical",
        "Physiological",
        "Physiological vs. Physiological 1000M",
        "10M vs. 1000M",
    ]
    data_pairs = [
        (results[0][0], results[0][1]),  # Physical
        (results[2][0], results[2][1]),  # Physiological
        (results[2][0], results[1][0]),  # Physiological vs. Physiological 1000M
        (results[2][0], results[3][0]),  # 10M vs. 1000M
    ]

    for label, (y_true, y_pred) in zip(labels, data_pairs, strict=False):
        print(f"\n===== Metrics for {label} data =====")

        # Compute the MAE
        ae = torch.abs(y_true - y_pred)
        ae_lower, ae_upper = compute_bootstrap_ci(ae.flatten().numpy(), 0.95)
        l1_loss = torch.nn.functional.l1_loss(y_pred, y_true).item()
        print(f"MAE: {ae.mean().item():.9f} (95% CI: [{ae_lower:.9f}, {ae_upper:.9f}])")
        print(f"L1 Loss: {l1_loss:.9f}")

        # Compute the standard error
        std_error_scipy = sem((y_true - y_pred).flatten())
        std = np.std((y_true - y_pred).flatten().numpy())
        print(f"Standard Deviation: {std:.9f}")
        print(f"Standard Error: {std / np.sqrt(len(y_true.flatten())):.9f}")
        print(f"Standard Error (scipy): {std_error_scipy:.9f}")

        # Compute the relative error
        rel_error = torch.abs(y_true - y_pred) / y_true * 100
        rel_error_lower, rel_error_upper = compute_bootstrap_ci(
            rel_error.flatten().numpy(), 0.95
        )
        print(
            f"Relative Error: {rel_error.mean().item():.9f}% "
            f"(95% CI: [{rel_error_lower:.9f}, {rel_error_upper:.9f}])"
        )

        # Compute the correlation coefficient
        corr = np.corrcoef(y_true, y_pred, rowvar=False)
        corr_len = corr.shape[0]
        _upper = corr[corr_len // 2 :, : corr_len // 2]
        _lower = corr[: corr_len // 2, corr_len // 2 :]
        print(f"Correlation Coefficient (_lower): {np.mean(_lower.diagonal()):.9f}")
        print(f"Correlation Coefficient (_upper): {np.mean(_upper.diagonal()):.9f}")
        corr_diag = custom_correlation_coefficient_diagonal(y_true, y_pred)
        print(f"Correlation Coefficient Diagonal: {corr_diag.mean().item():.9f}")

        # Compute mean and 95% CI for the mean for R^2
        r2 = custom_r2_score(y_true, y_pred)
        r2_lower, r2_upper = compute_bootstrap_ci(r2.flatten().numpy(), 0.95)
        print(
            f"R^2/CD^2: {r2.mean().item():.9f} "
            f"(95% CI: [{r2_lower:.9f}, {r2_upper:.9f}])"
        )
        # Compute R^2 using the legacy (old) method
        r2_old = compute_r2_old(
            y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy()
        )
        print(f"R^2/CD^2 (old): {r2_old:.9f}")

        adj_r2 = compute_adjusted_r2(y_true, y_pred, model)
        print(f"Adjusted R^2: {adj_r2:.9f}")


def create_visualization(
    y_physiological: torch.Tensor,
    y_physiological_1000M: torch.Tensor,
    y_pred_physiological: torch.Tensor,
) -> None:
    """Create and save the visualization plots."""
    # Prepare data
    reference = y_physiological.cpu().detach().numpy().flatten()
    reference_1000M = y_physiological_1000M.cpu().detach().numpy().flatten()
    rel_error = (
        np.abs(y_pred_physiological.cpu().detach().numpy().flatten() - reference_1000M)
        / reference_1000M
        * 100
    )
    rel_error_1000M = np.abs(reference - reference_1000M) / reference_1000M * 100

    # Compute theoretical bounds
    prob_grid = np.logspace(
        np.log10(np.min(reference)), np.log10(np.max(reference)), 10000
    )
    cov1mio = coeff_of_variation(prob_grid, 1e6) * 100
    cov10mio = coeff_of_variation(prob_grid, 1e7) * 100
    cov100mio = coeff_of_variation(prob_grid, 1e8) * 100
    lower_PI, upper_PI = rel_error_PI(prob_grid, int(1e7))
    lower_PI_100, upper_PI_100 = rel_error_PI(prob_grid, int(1e8))

    # Create plots
    fig, axs = plt.subplots(2, 1, figsize=(4, 4), sharex=True, sharey=True)
    bins_x = np.logspace(
        np.log10(np.min(reference) + 1e-8), np.log10(np.max(reference) + 1e-8), 100
    )
    bins_y = np.logspace(np.log10(1e-3), np.log10(80), 100)

    # Plot 1: Surrogate model predictions
    hb0 = axs[0].hist2d(
        reference + 1e-8,
        rel_error + 1e-8,
        bins=[bins_x, bins_y],
        norm=mcolors.LogNorm(),
        cmap="Blues",
    )
    fig.colorbar(hb0[3], ax=axs[0], label="Counts")
    axs[0].plot(
        prob_grid,
        cov1mio,
        color="red",
        linestyle="--",
        label="MLE: Coeff. of Variation (1M Photons)",
    )
    axs[0].plot(
        prob_grid,
        cov10mio,
        color="brown",
        linestyle="--",
        label="MLE: Coeff. of Variation (10M Photons)",
    )
    axs[0].fill_between(
        prob_grid,
        lower_PI,
        upper_PI,
        color="gray",
        alpha=0.3,
        label="95% Binomial PI (10M Photons)",
    )

    # Plot 2: Reference data
    custom_blue_cmap = LinearSegmentedColormap.from_list(
        "custom_blue", ["white", "darkblue"]
    )
    hb1 = axs[1].hist2d(
        reference_1000M + 1e-8,
        rel_error_1000M + 1e-8,
        bins=[bins_x, bins_y],
        norm=mcolors.LogNorm(),
        cmap=custom_blue_cmap,
    )
    fig.colorbar(hb1[3], ax=axs[1], label="Counts")
    axs[1].plot(
        prob_grid,
        cov10mio,
        color="brown",
        linestyle="--",
        label="MLE: Coeff. of Variation (10M Photons)",
    )
    axs[1].plot(
        prob_grid,
        cov100mio,
        color="black",
        linestyle="--",
        label="MLE: Coeff. of Variation (100M Photons)",
    )
    axs[1].fill_between(
        prob_grid,
        lower_PI_100,
        upper_PI_100,
        color="gray",
        alpha=0.3,
        label="MLE: Binomial 95% PI (100M Photons)",
    )

    # Configure axes
    for ax in axs:
        ax.set_ylim(1e-3, 80)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()

    axs[1].set_xlabel("Ground Truth Reflectance Values")
    axs[0].set_ylabel("Relative Error [%]")
    axs[1].set_ylabel("Relative Error [%]")
    axs[0].set_title("Relative Error of Surrogate Model Predictions")
    axs[1].set_title("Theoretical Against Empirical Lower Error Bound")

    # Add legend entries for histograms
    axs[0].plot([], [], "s", color="tab:blue", label="Rel. Error of Surrogate Model")
    axs[1].plot(
        [],
        [],
        "s",
        color=custom_blue_cmap(0.5),
        label="Empirical Lower Rel. Error Bound",
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            os.environ["plot_save_dir"], "model_error_distributions_physiological.svg"
        )
    )


def main() -> None:
    """Main function to run the model evaluation."""
    # Load configuration and model
    cfg = load_config()
    base_path = os.path.join(
        os.environ["data_dir"], cfg["surrogate"]["issi_model"]["base_path"]
    )
    checkpoint_path = cfg["surrogate"]["issi_model"]["checkpoint_path"]
    model_physical, preprocessor, model_cfg = load_trained_model(
        base_path, checkpoint_path, BaseModel
    )

    # Load and process all datasets
    (
        physical_data,
        physiological_data,
        physiological_1000M_data,
        physiological_10M_data,
    ) = load_and_process_data(preprocessor, model_cfg)

    logger.info(
        f"Dataset shapes: Physical {physical_data.shape}, "
        f"Physiological {physiological_data.shape}, "
        f"1000M {physiological_1000M_data.shape}, 10M {physiological_10M_data.shape}"
    )
    # Evaluate model on all datasets
    datasets = [
        (physical_data, "physical"),
        (physiological_data, "physiological"),
        (physiological_1000M_data, "physiological_1000M"),
        (physiological_10M_data, "physiological_10M"),
    ]
    results = evaluate_model(model_physical, datasets)

    compute_metrics(results, model_physical)

    create_visualization(results[1][0], results[2][0], results[1][1])


if __name__ == "__main__":
    main()
