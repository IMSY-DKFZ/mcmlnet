import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from mcmlnet.experiments.plotting import (
    add_theoretical_bounds,
    plot_scaling_fit,
    setup_plot_style,
)
from mcmlnet.experiments.utils import (
    compute_mae_ci,
    compute_reordering_array,
    neural_scaling_fit,
    neural_scaling_fit_log_scale,
)
from mcmlnet.training.data_loading.data_module import PreProcessor
from mcmlnet.utils.loading import SimulationDataLoader
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
        f"{physical_data.min().item():.10f}, {physical_data.max().item():.5f}"
    )
    logger.info(
        "Min/ Max values of the physiological data: "
        f"{physiological_data.min().item():.5f}, {physiological_data.max().item():.5f}"
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

    # CI estimation for the 100.000, 1.000.000, and 10.000.000 photon simulations
    cis = {}
    datasets = {
        100_000: "0_1M",
        1_000_000: "1M",
        10_000_000: "10M",
    }
    for n_photons in datasets.keys():
        # Load dataset and collect test data
        batch_folder = os.path.join(
            os.environ["data_dir"],
            "raw",
            "base_physio_and_physical_simulations",
            f"physiological_ablation_{datasets[n_photons]}_photons.parquet",
        )
        _data = pd.read_parquet(batch_folder)
        _data = torch.from_numpy(
            SimulationDataLoader().simulation_to_standard_numpy(_data)
        )
        # Apply correct data order and collect test data
        _test_data = _data[matched_indices]

        parameter_difference = torch.sum(
            physiological_data[:, :24] - _test_data[:, :24]
        )
        if parameter_difference > 1e-6:
            logger.warning(f"Summed Parameter Difference: {parameter_difference}")
        logger.info(
            f"Shape of the {n_photons / 10**6}M photon data: {_test_data.shape}"
        )
        logger.info(
            f"Min/ Max values of the {n_photons / 10**6}M photon data: "
            f"{_test_data.min().item():.5f}, {_test_data.max().item():.5f}"
        )

        _lower, _upper = compute_mae_ci(
            physiological_1000M_data[:, 24:].numpy(), _test_data[:, 24:].numpy()
        )
        cis[n_photons] = (_lower, _upper)

    # Add 100M photon data
    cis[100_000_000] = compute_mae_ci(
        physiological_1000M_data[:, 24:].numpy(),
        physiological_data[:, 24:].numpy(),
    )

    # MLE estimation
    mles = {}
    for n_photons in [100_000, 1_000_000, 10_000_000, 100_000_000]:
        _reflectance = physiological_1000M_data[:, 24:].flatten()

        logger.info(f"Dataset: Our Test Data - Reflectance Shape: {_reflectance.shape}")
        # Compute binomial distribution's std. dev. via the coefficient of variation
        _rel_error = coeff_of_variation(_reflectance, n_photons)
        _abs_error = _rel_error * _reflectance
        logger.info(
            f"Typical relative error for {int(n_photons)} photons: "
            f"{torch.mean(_rel_error) * 100:.3f} %"
        )
        logger.info(
            f"Typical absolute error for {int(n_photons)} photons: "
            f"{torch.mean(_abs_error):.3g}"
        )
        mles[n_photons] = torch.mean(_abs_error).item()

    # Visualization
    setup_plot_style()

    # Create a single plot for main_ax
    _fig, main_ax = plt.subplots(figsize=(6.5, 3))
    results_df = pd.read_csv(
        os.path.join(
            os.environ["cache_dir"], "100M_surrogate_model_kfold_10x_results.csv"
        )
    )
    sizes = results_df["tdr"].to_numpy() * 15 * 560000
    final_loss_100_000_000 = results_df["mae_test_physiological"].to_numpy()

    # Fit the polynomial function to the 100M photon test data
    x_fit, y_fit, y_fit_min, y_fit_max, popt, perr = neural_scaling_fit_log_scale(
        sizes, final_loss_100_000_000
    )
    plot_scaling_fit(
        main_ax,
        sizes,
        final_loss_100_000_000,
        popt,
        perr,
        label="Surrogate Model Test MAE (100M Photons)",
        color="tab:blue",
    )
    # Add theoretical bounds
    add_theoretical_bounds(main_ax, mles[100_000_000], cis[100_000_000])
    main_ax.set_xlabel("Training Dataset Size")
    main_ax.set_ylabel("Test Dataset MAE")
    main_ax.set_xscale("log")
    main_ax.set_yscale("log")
    # main_ax.set_ylim([3e-6, 4e-3])
    main_ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    main_ax.legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(os.environ["plot_save_dir"], "scaling_behaviour_main_ax_only.pdf")
    )

    # Create subplots with a 2x2 layout for the Appendix
    _fig, axes = plt.subplots(2, 2, figsize=(6.5, 4), sharex=True, sharey=True)

    # Define photon ablation data convergence manually
    results_100k_df = pd.read_csv(
        os.path.join(os.environ["cache_dir"], "100k_mlp_test_results.csv")
    )
    ablation_dataset_sizes_100k = results_100k_df["tdr"].to_numpy() * 15 * 560000
    final_loss_100_000 = results_100k_df["mae"].to_numpy()
    print(f"Final loss 100k: {final_loss_100_000}")

    results_1M_df = pd.read_csv(
        os.path.join(os.environ["cache_dir"], "1M_mlp_test_results.csv")
    )
    ablation_dataset_sizes_1M = results_1M_df["tdr"].to_numpy() * 15 * 560000
    final_loss_1_000_000 = results_1M_df["mae"].to_numpy()
    print(f"Final loss 1M: {final_loss_1_000_000}")

    results_10M_df = pd.read_csv(
        os.path.join(os.environ["cache_dir"], "10M_mlp_test_results.csv")
    )
    ablation_dataset_sizes_10M = results_10M_df["tdr"].to_numpy() * 15 * 560000
    final_loss_10_000_000 = results_10M_df["mae"].to_numpy()
    print(f"Final loss 10M: {final_loss_10_000_000}")

    # Manually shrink the legend
    plt.rcParams["legend.fontsize"] = 6

    # Fit and plot for 100K photons
    x_fit, y_fit, y_fit_min, y_fit_max, popt, perr = neural_scaling_fit_log_scale(
        ablation_dataset_sizes_100k, final_loss_100_000
    )
    plot_scaling_fit(
        axes[0, 0],
        ablation_dataset_sizes_100k,
        final_loss_100_000,
        popt,
        perr,
        label="Test Dataset MAE (0.1M Photons)",
        color="tab:blue",
    )
    add_theoretical_bounds(
        axes[0, 0], mles[100_000], cis[100_000], label_prefix="Theo. Lower MAE Bound"
    )
    axes[0, 0].set_ylabel("Test Dataset MAE")
    axes[0, 0].set_title("0.1M Photon Training Data Scaling")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlim([8e3, 1e8])
    axes[0, 0].set_ylim([1e-5, 1e-1])
    axes[0, 0].grid(True, which="major", linestyle="--", linewidth=1.0)
    axes[0, 0].grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.5)
    axes[0, 0].legend()

    # Fit and plot for 1M photons
    x_fit, y_fit, y_fit_min, y_fit_max, popt, perr = neural_scaling_fit_log_scale(
        ablation_dataset_sizes_1M,
        final_loss_1_000_000,
    )
    plot_scaling_fit(
        axes[0, 1],
        ablation_dataset_sizes_1M,
        final_loss_1_000_000,
        popt,
        perr,
        label="Test Dataset MAE (1M Photons)",
        color="tab:blue",
    )
    add_theoretical_bounds(
        axes[0, 1],
        mles[1_000_000],
        cis[1_000_000],
        label_prefix="Theo. Lower MAE Bound",
    )
    axes[0, 1].set_title("1M Photon Training Data Scaling")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlim([8e3, 1e8])
    axes[0, 1].set_ylim([1e-5, 1e-1])
    axes[0, 1].grid(True, which="major", linestyle="--", linewidth=1.0)
    axes[0, 1].grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.5)
    axes[0, 1].legend()

    # Fit and plot for 10M photons
    x_fit, y_fit, y_fit_min, y_fit_max, popt, perr = neural_scaling_fit_log_scale(
        ablation_dataset_sizes_10M,
        final_loss_10_000_000,
    )
    plot_scaling_fit(
        axes[1, 0],
        ablation_dataset_sizes_10M,
        final_loss_10_000_000,
        popt,
        perr,
        label="Test Dataset MAE (10M Photons)",
        color="tab:blue",
    )
    add_theoretical_bounds(
        axes[1, 0],
        mles[10_000_000],
        cis[10_000_000],
        label_prefix="Theo. Lower MAE Bound",
    )
    axes[1, 0].set_xlabel("Training Dataset Size")
    axes[1, 0].set_ylabel("Test Dataset MAE")
    axes[1, 0].set_title("10M Photon Training Data Scaling")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlim([8e3, 1e8])
    axes[1, 0].set_ylim([1e-5, 1e-1])
    axes[1, 0].grid(True, which="major", linestyle="--", linewidth=1.0)
    axes[1, 0].grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.5)
    axes[1, 0].legend()

    # Fit and plot for 100M photons
    x_fit, y_fit, y_fit_min, y_fit_max, popt, perr = neural_scaling_fit_log_scale(
        sizes, final_loss_100_000_000
    )
    plot_scaling_fit(
        axes[1, 1],
        sizes,
        final_loss_100_000_000,
        popt,
        perr,
        label="Test Dataset MAE (100M Photons)",
        color="tab:blue",
    )
    add_theoretical_bounds(
        axes[1, 1],
        mles[100_000_000],
        cis[100_000_000],
        label_prefix="Theo. Lower MAE Bound",
    )
    axes[1, 1].set_xlabel("Training Dataset Size")
    axes[1, 1].set_title("100M Photon Training Data Scaling")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlim([8e3, 1e8])
    axes[1, 1].set_ylim([1e-5, 1e-1])
    axes[1, 1].grid(True, which="major", linestyle="--", linewidth=1.0)
    axes[1, 1].grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.5)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            os.environ["plot_save_dir"], "scaling_behaviour_appendix_2x2_subplots.svg"
        )
    )

    # Create subplots with a 1x2 layout to replace the single main-ax figure
    _fig, axes = plt.subplots(1, 2, figsize=(6.5, 3), sharex=True, sharey=True)
    # Manually reset the legend font size
    plt.rcParams["legend.fontsize"] = 8

    # Fit and plot for 100K photons
    x_fit, y_fit, y_fit_min, y_fit_max, popt, perr = neural_scaling_fit_log_scale(
        ablation_dataset_sizes_100k, final_loss_100_000
    )
    plot_scaling_fit(
        axes[0],
        ablation_dataset_sizes_100k,
        final_loss_100_000,
        popt,
        perr,
        label="Surrogate Model Test MAE (0.1M Photons)",
        color="tab:blue",
    )
    add_theoretical_bounds(axes[0], mles[100_000], cis[100_000])
    axes[0].set_ylabel("Test Dataset MAE")
    axes[0].set_xlabel("Training Dataset Size (#Data Pairs)")
    axes[0].set_title("0.1M Photon Monte Carlo Simulations")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylim([3e-6, 1e-1])
    axes[0].grid(True, which="both", linestyle="--", linewidth=0.5)
    axes[0].legend()

    # Fit and plot for 100M photons
    x_fit, y_fit, y_fit_min, y_fit_max, popt, perr = neural_scaling_fit_log_scale(
        sizes, final_loss_100_000_000
    )
    plot_scaling_fit(
        axes[1],
        sizes,
        final_loss_100_000_000,
        popt,
        perr,
        label="Surrogate Model Test MAE (100M Photons)",
        color="tab:blue",
    )
    add_theoretical_bounds(axes[1], mles[100_000_000], cis[100_000_000])
    axes[1].set_xlabel("Training Dataset Size (#Data Pairs)")
    axes[1].set_title("100M Photon Monte Carlo Simulations")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_ylim([3e-6, 1e-1])
    axes[1].grid(True, which="both", linestyle="--", linewidth=0.5)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            os.environ["plot_save_dir"], "scaling_behaviour_appendix_1x2_subplots.pdf"
        )
    )

    # Define lower photon accuracy bound scaling behaviour manually
    lower_bounds = [
        cis[100_000][0],
        cis[100_000][1],
        cis[1_000_000][0],
        cis[1_000_000][1],
        cis[10_000_000][0],
        cis[10_000_000][1],
        cis[100_000_000][0],
        cis[100_000_000][1],
    ]
    photons_sims_in_M = (
        np.array(
            [
                0.1,
                0.1,
                1,
                1,
                10,
                10,
                100,
                100,
            ]
        )
        * 1e6
    )

    _fig, ax = plt.subplots(figsize=(6.5, 3))
    x_fit, y_fit, y_fit_min, y_fit_max, popt, perr = neural_scaling_fit(
        photons_sims_in_M, lower_bounds
    )
    # Manually reset the legend font size
    plt.rcParams["legend.fontsize"] = 8
    plot_scaling_fit(
        ax,
        photons_sims_in_M,
        lower_bounds,
        popt,
        perr,
        label="Lower Bound Scaling Behaviour",
        color="tab:blue",
        decimals={"a": 2, "c": 1},
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Photons Used in Monte Carlo Simulation")
    ax.set_ylabel("Lower Bound MAE (95% CI)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            os.environ["plot_save_dir"], "lower_bound_scaling_behaviour_appendix.svg"
        )
    )


if __name__ == "__main__":
    main()
