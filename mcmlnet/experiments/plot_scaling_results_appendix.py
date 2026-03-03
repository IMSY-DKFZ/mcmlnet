import os

import matplotlib.pyplot as plt
import pandas as pd
import torch

from mcmlnet.experiments.plotting import (
    add_theoretical_bounds,
    plot_scaling_fit,
    setup_plot_style,
)
from mcmlnet.experiments.utils import (
    compute_mae_ci,
    neural_scaling_fit,
    neural_scaling_fit_log_scale,
)
from mcmlnet.training.data_loading.data_module import PreProcessor
from mcmlnet.training.data_loading.sota_data_classes import SOTAPreprocessor
from mcmlnet.utils.loading import (
    SimulationDataLoader,
)
from mcmlnet.utils.logging import setup_logging
from mcmlnet.utils.process_spectra import coeff_of_variation

logger = setup_logging(level="info", logger_name=__name__)

if not os.path.exists(os.environ["plot_save_dir"]):
    os.makedirs(os.environ["plot_save_dir"])


def main() -> None:
    # Load datasets
    physical_data = SimulationDataLoader().load_physical_simulation_data(
        "raw/base_physio_and_physical_simulations/physical_generalization_100M_photons.parquet",
        n_wavelengths=351,
    )
    physiological_data = SimulationDataLoader().load_simulation_data(
        "raw/base_physio_and_physical_simulations/physiological_training_100M_photons_1M_samples.parquet",
    )
    # Allocate test dataset(s)
    preprocessor = PreProcessor(val_percent=0.1, test_percent=0.2)
    physical_data = physical_data[
        preprocessor.consistent_data_split_ids(physical_data, mode="test")
    ]
    physiological_data = physiological_data[
        preprocessor.consistent_data_split_ids(physiological_data, mode="test")
    ]
    logger.info(f"Shape of the physical data: {physical_data.shape}")
    logger.info(f"Shape of the physiological data: {physiological_data.shape}")
    logger.info(
        "Min/ Max values of the physical data: "
        f"{physical_data.min().item():.10f}, {physical_data.max():.5f}"
    )
    logger.info(
        "Min/ Max values of the physiological data: "
        f"{physiological_data.min():.5f}, {physiological_data.max():.5f}"
    )
    # Manually load 1 billion photon dataset
    batch_folder = os.path.join(
        os.environ["data_dir"],
        "raw/base_physio_and_physical_simulations/physiological_training_100M_1M_samples_test_10x_photons.parquet",
    )
    physiological_1000M_df = pd.read_parquet(batch_folder)
    physiological_1000M_data = torch.from_numpy(
        SimulationDataLoader().simulation_to_standard_numpy(physiological_1000M_df)
    )
    logger.info(f"Shape of the 1000M photon data: {physiological_1000M_data.shape}")
    logger.info(
        "Min/ Max values of the 1000M physiological data: "
        f"{physiological_1000M_data.min().item():.5f}, "
        f"{physiological_1000M_data.max().item():.5f}"
    )

    # CI estimation
    cis = {
        100_000_000: compute_mae_ci(
            physiological_1000M_data[:, 24:].numpy(),
            physiological_data[:, 24:].numpy(),
        )
    }

    # MLE estimation
    mles = {}
    for _dataset in ["lan_lhs", "tsui", "manoj"]:
        if _dataset == "manoj":
            n_wavelengths = 351
            n_photons = 10**6
        else:
            n_wavelengths = 1
            n_photons = 10**8
        _preprocessor = SOTAPreprocessor(
            _dataset,
            n_wavelengths,
        )
        _test_ids = _preprocessor.consistent_data_split_ids(
            _preprocessor.labels, "test"
        )
        _test_reflectance = _preprocessor.labels[_test_ids].flatten()
        logger.info(
            f"Dataset: {_dataset} - Reflectance Shape: {_test_reflectance.shape}"
        )
        # Compute binomial distribution's std. dev. via the coefficient of variation
        _rel_error = coeff_of_variation(_test_reflectance, n_photons)
        _abs_error = _rel_error * _test_reflectance
        logger.info(
            f"Typical relative error for {int(n_photons)} photons: "
            f"{torch.mean(_rel_error) * 100:.3f} %"
        )
        logger.info(
            f"Typical absolute error for {int(n_photons)} photons: "
            f"{torch.mean(_abs_error):.3g}"
        )
        mles[_dataset] = torch.mean(_abs_error).item()

    # Add KAN surrogate model
    _rel_error = coeff_of_variation(
        physiological_1000M_data[:, 24:].flatten(), 100_000_000
    )
    _abs_error = _rel_error * physiological_1000M_data[:, 24:].flatten()
    mles["kan"] = torch.mean(_abs_error).item()

    # Visualization
    setup_plot_style()

    # Define manual related work data
    results_kan_df = pd.read_csv(
        os.path.join(os.environ["cache_dir"], "100M_kan_10x_results.csv")
    )
    kan_dataset_sizes = results_kan_df["tdr"].to_numpy() * 15 * 560000
    kan_final_losses = results_kan_df["mae_test_physiological"].to_numpy()

    results_tsui_df = pd.read_csv(
        os.path.join(os.environ["cache_dir"], "tsui_results.csv")
    )
    scaling_factors = results_tsui_df["tdr"].to_numpy()
    dataset_sizes_tsui = scaling_factors * 21000
    final_loss_tsui = results_tsui_df["mae"].to_numpy()

    results_lan_df = pd.read_csv(
        os.path.join(os.environ["cache_dir"], "lan_results.csv")
    )
    dataset_sizes_lan = scaling_factors * 3500
    final_loss_lan = results_lan_df["mae"].to_numpy()

    results_manojlovic_df = pd.read_csv(
        os.path.join(os.environ["cache_dir"], "manoj_results.csv")
    )
    dataset_sizes_manojlovic = results_manojlovic_df["tdr"].to_numpy() * 24500 * 351
    final_loss_manojlovic = results_manojlovic_df["mae"].to_numpy()

    # Create subplots with a 2x2 layout
    _fig, axes = plt.subplots(2, 2, figsize=(6.5, 4), sharey=False)

    # Fit the polynomial function to the 100M photon KAN surrogate model data
    x_fit, y_fit, y_fit_min, y_fit_max, popt, perr = neural_scaling_fit(
        kan_dataset_sizes, kan_final_losses
    )
    # Manually shrink the legend
    plt.rcParams["legend.fontsize"] = 6
    plot_scaling_fit(
        axes[0, 0],
        kan_dataset_sizes,
        kan_final_losses,
        popt,
        perr,
        "KAN Surrogate Model Test MAE (100M Photons)",
        "tab:blue",
        decimals={"a": -1, "c": 1},
    )
    add_theoretical_bounds(
        axes[0, 0], mles["kan"], cis[100_000_000], label_prefix="Theo. Lower MAE Bound"
    )
    axes[0, 0].set_ylabel("Test Dataset MAE")
    axes[0, 0].set_title("KAN Photon Training Data Scaling")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylim([1e-5, 1e-1])
    axes[0, 0].grid(True, which="major", linestyle="--", linewidth=1.0)
    axes[0, 0].grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.5)
    axes[0, 0].legend()

    # Fit the polynomial function to the Lan et al. (2023) data
    x_fit, y_fit, y_fit_min, y_fit_max, popt, perr = neural_scaling_fit_log_scale(
        dataset_sizes_lan, final_loss_lan
    )
    plot_scaling_fit(
        axes[0, 1],
        dataset_sizes_lan,
        final_loss_lan,
        popt,
        perr,
        label="Lan et al. (2023) Test MAE (100M Photons)",
        color="tab:purple",
        decimals={"a": 1, "c": 1},
    )
    add_theoretical_bounds(
        axes[0, 1], mles["lan_lhs"], label_prefix="Theo. Lower MAE Bound"
    )
    axes[0, 1].set_title("Lan et al. (2023) Training Data Scaling")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylim([1e-5, 1e-1])
    axes[0, 1].grid(True, which="major", linestyle="--", linewidth=1.0)
    axes[0, 1].grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.5)
    axes[0, 1].legend()

    # Fit the polynomial function to the Tsui et al. (2018) data
    x_fit, y_fit, y_fit_min, y_fit_max, popt, perr = neural_scaling_fit_log_scale(
        dataset_sizes_tsui, final_loss_tsui
    )
    plot_scaling_fit(
        axes[1, 0],
        dataset_sizes_tsui,
        final_loss_tsui,
        popt,
        perr,
        label="Tsui et al. (2018) Test MAE (100M Photons)",
        color="tab:green",
        decimals={"a": 1, "c": 1},
    )
    add_theoretical_bounds(
        axes[1, 0], mles["tsui"], label_prefix="Theo. Lower MAE Bound"
    )
    axes[1, 0].set_xlabel("Training Dataset Size")
    axes[1, 0].set_ylabel("Test Dataset MAE")
    axes[1, 0].set_title("Tsui et al. (2018) Training Data Scaling")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylim([1e-5, 1e-1])
    axes[1, 0].grid(True, which="major", linestyle="--", linewidth=1.0)
    axes[1, 0].grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.5)
    axes[1, 0].legend()

    # Fit the polynomial function to the Manojlovic et al. (2025) data
    x_fit, y_fit, y_fit_min, y_fit_max, popt, perr = neural_scaling_fit_log_scale(
        dataset_sizes_manojlovic, final_loss_manojlovic
    )
    plot_scaling_fit(
        axes[1, 1],
        dataset_sizes_manojlovic,
        final_loss_manojlovic,
        popt,
        perr,
        label="Manojlovic et al. (2025) Test MAE (1M Photons)",
        color="tab:orange",
        decimals={"a": -1, "c": 1},
    )
    add_theoretical_bounds(
        axes[1, 1], mles["manoj"], label_prefix="Theo. Lower MAE Bound"
    )
    axes[1, 1].set_xlabel("Training Dataset Size")
    axes[1, 1].set_title("Manojlovic et al. (2025) Training Data Scaling")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylim([1e-5, 1e-1])
    axes[1, 1].grid(True, which="major", linestyle="--", linewidth=1.0)
    axes[1, 1].grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.5)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            os.environ["plot_save_dir"],
            "scaling_behaviour_sota_appendix_2x2_subplots.svg",
        )
    )


if __name__ == "__main__":
    main()
