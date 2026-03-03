"""
Evaluate absorption-scattering surface and Shapley coefficients
of the forward surrogate model.

Features:
- 3D surface plotting of reflectance predictions
- SHAP-based model explainability analysis
- Ensemble model handling for SHAP compatibility
- Plot saving and formatting utilities
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import torch
from dotenv import load_dotenv
from matplotlib.colors import SymLogNorm
from rich.progress import track

from mcmlnet.experiments.plotting import setup_plot_style
from mcmlnet.susi.calculate_spectra import calculate_spectrum_for_physical_batch
from mcmlnet.training.data_loading.preprocessing import PreProcessor
from mcmlnet.training.models.base_model import BaseModel
from mcmlnet.utils.caching import np_cache_to_file
from mcmlnet.utils.convenience import load_trained_model, predict_in_batches
from mcmlnet.utils.load_configs import load_config
from mcmlnet.utils.loading import SimulationDataLoader
from mcmlnet.utils.logging import setup_logging

load_dotenv()

# Setup logging
logger = setup_logging(level="info", logger_name=__name__)


# Data configuration
DATA_PATH: str = (
    "raw/base_physio_and_physical_simulations/"
    "physiological_training_100M_photons_1M_samples.parquet"
)
N_WAVELENGTHS: int = 15
N_LAYERS: int = 3

# SHAP analysis configuration
N_SAMPLES: int = 750
BATCH_SIZE: int = 50
SHAP_BATCH_SIZE: int = 10

# Plotting configuration
SAVE_PREFIX: str = "MLP_SHAP_"
PLOT_SAVE_DIR: str | None = None

# Analysis configuration
SUBSAMPLE_FACTOR: int = 10
RANDOM_SEED: int = 42

if not os.path.exists(os.environ["plot_save_dir"]):
    os.makedirs(os.environ["plot_save_dir"])


# =============================================================================
# Parameter and Data Utilities
# =============================================================================


def get_param_names(n_layers: int = 3) -> list[str]:
    """Generate parameter names for physical parameters in LaTeX format.

    Args:
        n_layers: Number of tissue layers. Defaults to 3.

    Returns:
        List of parameter names in LaTeX format for absorption coefficient,
        scattering coefficient, anisotropy factor, refractive index, and thickness.
    """
    param_names = []
    for name in [r"\mu_{a,", r"\mu_{s,", "g_{", "n_{", "d_{"]:
        for i in range(n_layers):
            param_names.append(r"$%s%d}$" % (name, i + 1))
    return param_names


# =============================================================================
# Visualization Functions
# =============================================================================


def plot_3d_surface(
    mu_a: torch.Tensor,
    mu_s: torch.Tensor,
    reflectance: torch.Tensor,
    symlog_z: bool = False,
) -> plt.Figure:
    """Plot a 3D surface visualization of reflectance predictions.

    Args:
        mu_a: Absorption coefficient values for x-axis
        mu_s: Scattering coefficient values for y-axis
        reflectance: Predicted reflectance values for z-axis
        symlog_z: Whether to use symlog scaling for z-axis. Defaults to False

    Returns:
        The matplotlib figure object
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Convert tensors to numpy for plotting
    x = mu_a.cpu().numpy()
    y = mu_s.cpu().numpy()
    z = reflectance.cpu().numpy()

    # Create surface plot
    if not symlog_z:
        mesh = ax.plot_trisurf(x, y, z, cmap=plt.get_cmap("RdBu_r"), vmin=0, vmax=1)
    else:
        mesh = ax.plot_trisurf(x, y, z, cmap=plt.get_cmap("RdBu_r"))

    # Add colorbar and labels
    plt.colorbar(mesh, label="Predicted Reflectance")
    ax.set_xlabel("$\\mu_{{a,1}}$")
    ax.set_ylabel("$\\mu_{{s,1}}$")
    ax.set_zlabel("Reflectance")
    ax.zaxis.labelpad = -0.7

    # Apply symlog scaling if requested
    if symlog_z:
        ax.set_zscale("symlog")
        ax.set_zlabel("Logarithmic Reflectance")

    plt.title("Reflectance Prediction Resulting from $\\mu_{{a,1}}$ and $\\mu_{{s,1}}$")
    plt.tight_layout()

    return fig


def save_plot_notebook(save_str: str) -> None:
    """Save the current matplotlib plot as both PNG and PDF with specified formatting.

    The PDF is saved with a maximum width of 6.5 inches while maintaining aspect ratio.
    The PNG is saved at 300 DPI for high quality.

    Args:
        save_str: Filename (without extension) to save the plots
    """
    fig = plt.gcf()
    current_size = fig.get_size_inches()
    width, height = current_size

    # Scale down if width exceeds 6.5 inches
    if width > 6.5:
        scale_factor = 6.5 / width
        new_width = 6.5
        new_height = height * scale_factor
        fig.set_size_inches(new_width, new_height)

    # Save as PDF with width constraint
    plt.savefig(
        os.path.join(os.environ["plot_save_dir"], save_str.replace(".png", ".pdf")),
        dpi=900,
        format="pdf",
        bbox_inches="tight",
    )

    # Restore original size and save as PNG
    fig.set_size_inches(current_size)
    plt.savefig(
        os.path.join(os.environ["plot_save_dir"], save_str),
        dpi=300,
        format="png",
        bbox_inches="tight",
    )
    plt.close()


def generate_surface_and_contour_plots(
    model: BaseModel,
    preprocessor: PreProcessor,
    val_normalized: torch.Tensor,
    reflectance: torch.Tensor,
) -> None:
    """Generate surface and contour plots for different data scenarios.

    This function creates plots for:
    - Clean data (grid-based predictions)
    - Model predictions on real parameters
    - Real reflectance values
    - Difference between predicted and real values

    Args:
        model: Trained model for predictions
        preprocessor: Data preprocessor for normalization
        val_normalized: Normalized validation data
        reflectance: Ground truth reflectance values
    """

    # Helper function to undo normalization
    def undo_normalization(
        data: np.ndarray, preprocessor: PreProcessor, col: int
    ) -> np.ndarray:
        """Undo normalization to get original parameter values."""
        std = preprocessor.norm_2.cpu().numpy()[col]  # type: ignore [union-attr]
        mean = preprocessor.norm_1.cpu().numpy()[col]  # type: ignore [union-attr]

        return data * std + mean

    # Helper function to get clean (median) data grid
    def get_clean_data(
        reference_data: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate clean grid data for surface plotting."""
        # Create a grid based on the parameter ranges
        mu_a_1_min, mu_a_1_max = reference_data[:, 0].min(), reference_data[:, 0].max()
        mu_s_1_min, mu_s_1_max = reference_data[:, 3].min(), reference_data[:, 3].max()

        # Create grid
        mu_a_1_grid = np.linspace(mu_a_1_min, mu_a_1_max, 2000)
        mu_s_1_grid = np.linspace(mu_s_1_min, mu_s_1_max, 2000)
        mu_a_1_mesh, mu_s_1_mesh = np.meshgrid(mu_a_1_grid, mu_s_1_grid)

        # Create clean data array with other parameters set to median values
        clean_data = np.zeros((mu_a_1_mesh.size, reference_data.shape[1]))
        for i in range(reference_data.shape[1]):
            if i == 0:  # mu_a_1
                clean_data[:, i] = mu_a_1_mesh.ravel()
            elif i == 3:  # mu_s_1
                clean_data[:, i] = mu_s_1_mesh.ravel()
            else:  # Other parameters set to median
                clean_data[:, i] = np.median(reference_data[:, i])

        return clean_data, mu_a_1_mesh, mu_s_1_mesh

    # Get data for plotting
    shap_physical_data_np = (
        val_normalized[::SUBSAMPLE_FACTOR]
        .reshape(-1, N_WAVELENGTHS)
        .detach()
        .cpu()
        .numpy()
    )

    # Run model reflectance estimation on clean data
    clean_data, mu_a_1_grid, mu_s_1_grid = get_clean_data(
        shap_physical_data_np,
    )
    mu_a_1_grid_unnorm = undo_normalization(mu_a_1_grid, preprocessor, 0)
    mu_s_1_grid_unnorm = undo_normalization(mu_s_1_grid, preprocessor, 3)

    clean_pred = (
        predict_in_batches(
            model,
            torch.from_numpy(clean_data).float().cuda(),
            batch_size=1000,
        )
        .detach()
        .cpu()
        .squeeze()
        .numpy()
    )

    clean_grid = [mu_a_1_grid_unnorm.ravel(), mu_s_1_grid_unnorm.ravel(), clean_pred]

    # Run model reflectance estimation on real parameters
    model_pred = (
        predict_in_batches(
            model,
            torch.from_numpy(shap_physical_data_np).float().cuda(),
            batch_size=1000,
        )
        .detach()
        .cpu()
        .squeeze()
        .numpy()
    )
    mu_a_1 = shap_physical_data_np[:, 0]
    mu_s_1 = shap_physical_data_np[:, 3]

    model_grid = [
        undo_normalization(mu_a_1, preprocessor, 0),
        undo_normalization(mu_s_1, preprocessor, 3),
        model_pred,
    ]

    # Take the real data
    real_grid = [
        model_grid[0],
        model_grid[1],
        reflectance[::SUBSAMPLE_FACTOR].flatten().detach().cpu().numpy(),
    ]

    # Show the difference
    diff_grid = [
        model_grid[0],
        model_grid[1],
        (model_grid[2] - real_grid[2]),
    ]

    # Generate plots for each data type
    plot_names = ["clean", "predicted", "real", "diff"]
    plot_data = [clean_grid, model_grid, real_grid, diff_grid]

    for idx, (data, name) in enumerate(zip(plot_data, plot_names, strict=False)):
        if idx == 0:
            # Plot clean data as surface plot using plotly
            fig = go.Figure(
                data=[
                    go.Surface(
                        z=clean_pred.reshape((len(mu_a_1_grid), len(mu_s_1_grid))),
                        x=mu_a_1_grid_unnorm,
                        y=mu_s_1_grid_unnorm,
                        hovertemplate=(
                            "µ<sub>a,1</sub>: %{x:.3f}<br>µ<sub>s,1</sub>: "
                            "%{y:.3f}<br>Predicted Reflectance: %{z:.3f}<extra></extra>"
                        ),
                    )
                ]
            )
            fig.update_traces(
                colorscale="RdBu_r",
                colorbar={"title": "Predicted Reflectance"},
                contours_z={
                    "show": True,
                    "usecolormap": True,
                    "highlightcolor": "limegreen",
                    "project_z": True,
                },
            )
            fig.update_layout(
                title=(
                    "Reflectance Prediction Resulting from µ<sub>a,1</sub> "
                    "and µ<sub>s,1</sub>"
                ),
                scene={
                    "xaxis_title": "µ<sub>a,1</sub>",
                    "yaxis_title": "µ<sub>s,1</sub>",
                    "zaxis_title": "Refl.",
                },
            )
            fig.write_html(
                os.path.join(
                    os.environ["plot_save_dir"], f"{SAVE_PREFIX}2d_{name}_surface.html"
                )
            )

        # Create contour plots for all data types
        plt.figure(figsize=(10, 6))
        if idx == 3:  # Difference plot uses symlog normalization
            norm = SymLogNorm(
                linthresh=1e-4, linscale=1.0, vmin=np.min(data[2]), vmax=np.max(data[2])
            )
            mesh = plt.tricontourf(
                data[0],
                data[1],
                data[2],
                levels=100,
                cmap=plt.get_cmap("RdBu_r"),
                norm=norm,
            )
        else:
            mesh = plt.tricontourf(
                data[0],
                data[1],
                data[2],
                levels=np.linspace(0, 1, 100),
                cmap=plt.get_cmap("RdBu_r"),
            )

        # Add a colorbar with explicit limits
        plt.colorbar(mesh, label="Predicted Reflectance")
        plt.title(
            "Reflectance Prediction Resulting from $\\mu_{{a,1}}$ and $\\mu_{{s,1}}$"
        )
        plt.xlabel("log $\\mu_{{a,1}}$")
        plt.ylabel("log $\\mu_{{s,1}}$")
        plt.tight_layout()

        # Save the plot
        plt.savefig(
            os.path.join(
                os.environ["plot_save_dir"], f"{SAVE_PREFIX}2d_{name}_tricontourf.png"
            ),
            dpi=600,
        )
        plt.close()


# =============================================================================
# SHAP Analysis Utilities
# =============================================================================


class EnsembleModel(torch.nn.Module):
    """Ensemble model wrapper for SHAP compatibility.

    This class wraps multiple BaseModel instances into a single nn.Module that
    SHAP can use. It handles normalization and averaging of predictions across
    ensemble members.

    Args:
        models: List of trained model instances
        mean: List of normalization means for each model
        std: List of normalization standard deviations for each model
    """

    def __init__(
        self,
        models: list[BaseModel],
        mean: list[torch.Tensor],
        std: list[torch.Tensor],
    ):
        super().__init__()
        self.models = models
        self.mean = mean
        self.std = std

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ensemble model.

        Args:
            data: Input data tensor

        Returns:
            Averaged predictions from all ensemble members
        """
        y_preds = []
        for i, model in enumerate(self.models):
            normalized_data = (data.cuda() - self.mean[i].cuda().unsqueeze(0)) / (
                self.std[i].cuda() + 1e-8
            )
            y_preds.append(model(normalized_data))

        return torch.mean(torch.cat(y_preds, dim=1), dim=-1, keepdim=True)


def shapley_values_predict(
    data: torch.Tensor | np.ndarray, explainer: shap.Explainer, batch_size: int = 10
) -> np.ndarray:
    """Calculate SHAP values for input data using batch processing.

    Args:
        data: Input data for which to calculate SHAP values
        explainer: Pre-configured SHAP explainer instance
        batch_size: Number of samples to process per batch. Defaults to 10

    Returns:
        Concatenated SHAP values for all input samples
    """
    shap_list = []
    n_batches = len(data) // batch_size

    for i in track(range(n_batches), description="Batch Progress"):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size

        if isinstance(data, torch.Tensor):
            batch_data = data[start_idx:end_idx]
        elif isinstance(data, np.ndarray):
            batch_data = data[start_idx:end_idx]
        else:
            raise TypeError("Data must be a torch.Tensor or np.ndarray.")

        shaps = explainer.shap_values(batch_data)

        if isinstance(shaps, torch.Tensor):
            shaps = shaps.detach().cpu().numpy()
        shap_list.append(shaps)

    return np.concatenate(shap_list, axis=0)


def html_force_plot(
    data: np.ndarray,
    feature_names: list[str],
    explainer: shap.Explainer,
    shap_vals: np.ndarray,
    save_str: str,
    sample_ind: int = 50,
) -> None:
    """Generate and save an interactive HTML force plot for SHAP values.

    Args:
        data: Input data features
        feature_names: Names of the features for labeling
        explainer: SHAP explainer instance with expected values
        shap_vals: Calculated SHAP values
        save_str: String to append to the filename for saving
        sample_ind: Number of samples to include in the plot. Defaults to 50
    """
    force_plot = shap.force_plot(
        base_value=explainer.expected_value[0],
        shap_values=shap_vals[:sample_ind],
        features=data[:sample_ind],
        feature_names=feature_names,
    )

    save_path = os.path.join(
        os.environ["plot_save_dir"], save_str + "force_plot_physical.html"
    )
    shap.plots._force.save_html(save_path, force_plot, full_html=True)


# =============================================================================
# Main Analysis Pipeline
# =============================================================================


def run_base_shap_analysis(
    model: BaseModel,
    val_normalized: torch.Tensor,
) -> None:
    """Run complete SHAP analysis pipeline for model explainability.

    Args:
        model: Trained model for analysis
        val_normalized: Normalized validation data
    """
    # Setup plotting style
    setup_plot_style()

    # Get parameter names and save string
    param_names = get_param_names(N_LAYERS)

    # Prepare and reduce data for SHAP analysis
    shap_physical_data = val_normalized[::SUBSAMPLE_FACTOR].reshape(-1, N_WAVELENGTHS)
    explainer = shap.DeepExplainer(model, shap_physical_data.cuda())

    # Select subset for analysis
    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.choice(shap_physical_data.shape[0], N_SAMPLES, replace=False)
    subset = shap_physical_data[indices]
    logger.info(f"Data subset shape: {subset.shape}")

    # Compute SHAP values
    shap_values = np_cache_to_file(
        shapley_values_predict,
        "shap/physical_surrogate_model_shap_values.npz",
    )(subset, explainer, batch_size=len(subset) // SHAP_BATCH_SIZE)
    logger.info(f"SHAP values shape: {shap_values.shape}")

    # Generate summary plot
    shap.summary_plot(
        shap_values.squeeze(),
        subset.cpu().numpy(),
        feature_names=param_names,
        show=False,
    )
    save_plot_notebook(f"{SAVE_PREFIX}summary_physical_mlp.png")

    # Generate partial dependence plots
    for param_idx, param_name in [(0, "absorption"), (3, "scattering")]:
        shap.partial_dependence_plot(
            param_names[param_idx],
            lambda x: predict_in_batches(
                model, torch.from_numpy(x).float().cuda(), batch_size=1000
            )
            .cpu()
            .numpy(),
            subset.cpu().numpy(),
            feature_names=param_names,
            model_expected_value=True,
            feature_expected_value=True,
            ice=False,
            show=False,
        )
        save_plot_notebook(f"{SAVE_PREFIX}partial_dependence_physical_{param_name}.png")

    # Generate waterfall and heatmap plots
    shap_waterfall_info = shap.Explanation(
        base_values=explainer.expected_value,
        values=shap_values[0].squeeze(),
        feature_names=param_names,
        data=subset[0].cpu().numpy(),
    )
    shap.plots.waterfall(shap_waterfall_info, max_display=15, show=False)
    plt.tight_layout()
    save_plot_notebook(f"{SAVE_PREFIX}waterfall_physical.png")

    shap_waterfall_infos = shap.Explanation(
        base_values=explainer.expected_value,
        values=shap_values.squeeze(),
        feature_names=param_names,
        data=subset.cpu().numpy(),
    )
    shap.plots.heatmap(shap_waterfall_infos, max_display=15, show=False)
    save_plot_notebook(f"{SAVE_PREFIX}heatmap_physical.png")

    # Generate HTML force plot
    html_force_plot(
        subset.cpu().numpy(), param_names, explainer, shap_values.squeeze(), SAVE_PREFIX
    )

    # Ablate the DeepExplainer dataset size to check if it affects the results
    explainer_ablated = shap.DeepExplainer(model, shap_physical_data[::5].cuda())
    # Compute SHAP values
    shap_values_ablated = np_cache_to_file(
        shapley_values_predict,
        "shap/physical_surrogate_model_shap_values_ablated.npz",
    )(subset, explainer_ablated, batch_size=len(subset) // SHAP_BATCH_SIZE)
    logger.info(f"SHAP values shape: {shap_values_ablated.shape}")
    # Compute the difference between the original and ablated SHAP values
    shap_diff_ablated = (shap_values_ablated - shap_values).squeeze()
    logger.info(f"SHAP values difference shape: {shap_diff_ablated.shape}")
    logger.info(f"SHAP values difference max: {shap_diff_ablated.max()}")
    logger.info(f"SHAP values difference min: {shap_diff_ablated.min()}")
    logger.info(f"SHAP values difference mean: {shap_diff_ablated.mean()}")
    logger.info(f"SHAP values difference median: {np.median(shap_diff_ablated)}")
    # Generate summary plot
    shap.summary_plot(
        shap_diff_ablated,
        subset.cpu().numpy(),
        feature_names=param_names,
        show=False,
    )
    save_plot_notebook(f"{SAVE_PREFIX}summary_physical_mlp_ablated.png")


def analyze_model_performance(
    reflectance: torch.Tensor,
    predicted_reflectance: torch.Tensor,
) -> np.ndarray:
    """Analyze model performance and generate performance plots.

    Args:
        reflectance: Ground truth reflectance values
        predicted_reflectance: Model predicted reflectance values

    Returns:
        Array of indices of the best, median, and worst cases
    """
    # Calculate normalized MAE
    nmae = (
        torch.abs((reflectance - predicted_reflectance).flatten())
        / reflectance.flatten()
    )

    # Identify best, median, and worst cases
    worst_5_ids = torch.argsort(nmae)[-5:]
    top_5_ids = torch.argsort(nmae)[:5]
    median_5_ids = torch.argsort(nmae)[len(nmae) // 2 - 2 : len(nmae) // 2 + 3]

    # Plot normalized MAE distribution
    bins = np.logspace(
        np.log10(nmae.min().item() * 100 + 1e-8),
        np.log10(nmae.max().item() * 100),
        100,
    )
    plt.figure(figsize=(10, 6))
    plt.hist(nmae.cpu().numpy() * 100, bins=bins)
    plt.xlabel("Normalized MAE [%]")
    plt.ylabel("Counts")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True)
    plt.axvline(
        nmae.mean().item() * 100,
        color="red",
        linestyle="--",
        label=f"Mean: {nmae.mean().item() * 100:.2f}%",
    )
    plt.axvline(
        nmae.median().item() * 100,
        color="green",
        linestyle="--",
        label=f"Median: {nmae.median().item() * 100:.2f}%",
    )
    plt.legend()
    plt.title("Normalized MAE of Reflectance Prediction")
    plt.tight_layout()
    save_plot_notebook(f"{SAVE_PREFIX}nmae_distribution.png")

    # Print performance summary
    print("Performance Summary:")
    print(f"Best cases (IDs: {top_5_ids.tolist()})")
    print(f"Median cases (IDs: {median_5_ids.tolist()})")
    print(f"Worst cases (IDs: {worst_5_ids.tolist()})")

    print("\nReflectance Values (Predicted vs. True):")
    print("Best cases:", predicted_reflectance.flatten()[top_5_ids].numpy())
    print("True values:", reflectance.flatten()[top_5_ids].numpy())
    print("Median cases:", predicted_reflectance.flatten()[median_5_ids].numpy())
    print("True values:", reflectance.flatten()[median_5_ids].numpy())
    print("Worst cases:", predicted_reflectance.flatten()[worst_5_ids].numpy())
    print("True values:", reflectance.flatten()[worst_5_ids].numpy())

    return np.concatenate((top_5_ids, median_5_ids, worst_5_ids))


def run_case_shap_analysis(
    model: BaseModel,
    val_normalized: torch.Tensor,
    case_indices: np.ndarray,
) -> np.ndarray:
    """Run SHAP analysis for best, median, and worst cases.

    Args:
        model: Trained model for analysis
        val_normalized: Normalized validation data
        case_indices: Indices of the best, median, and worst cases
    """
    # Setup plotting style
    setup_plot_style()

    # Get parameter names and save string
    param_names = get_param_names(N_LAYERS)
    case_mapping = ["Best Case"] * 5 + ["Median Case"] * 5 + ["Worst Case"] * 5

    # Prepare and reduce data for SHAP analysis
    shap_physical_data = val_normalized[::SUBSAMPLE_FACTOR].reshape(-1, N_WAVELENGTHS)
    explainer = shap.DeepExplainer(model, shap_physical_data.cuda())

    # Select subset for analysis
    subset = val_normalized.reshape(-1, 15)[case_indices]
    logger.info(f"Data subset shape: {subset.shape}")

    # Compute SHAP values
    shap_values_mlp = shapley_values_predict(subset, explainer, batch_size=len(subset))
    logger.info(f"SHAP values shape: {shap_values_mlp.shape}")

    # Generate summary plot
    shap.summary_plot(
        shap_values_mlp.squeeze(),
        subset.cpu().numpy(),
        feature_names=param_names,
        show=False,
    )
    save_plot_notebook(f"{SAVE_PREFIX}summary_physical_mlp_cases.png")

    for i, idx in track(enumerate(case_indices), description="Case Progress"):
        shap_waterfall_info = shap.Explanation(
            base_values=explainer.expected_value,
            values=shap_values_mlp[i].squeeze(),
            feature_names=param_names,
            data=subset[i].cpu().numpy(),
        )
        shap.plots.waterfall(shap_waterfall_info, max_display=15, show=False)
        plt.tight_layout()
        plt.title(f"Waterfall Plot for {case_mapping[i]} {i % len(case_mapping)}")
        save_plot_notebook(
            f"{SAVE_PREFIX}waterfall_physical_mlp_{case_mapping[i]}_{idx}.png"
        )

    return shap_values_mlp


def run_case_shap_analysis_monte_carlo(
    val_unnormalized: torch.Tensor,
    case_indices: np.ndarray,
    reflectance: torch.Tensor,
) -> np.ndarray:
    """
    Run SHAP analysis for best, median, and worst cases
    using ground truth Monte Carlo simulation.

    Args:
        val_unnormalized: Unnormalized validation data
        case_indices: Indices of the best, median, and worst cases
        reflectance: Ground truth reflectance values

    Returns:
        SHAP values for the best, median, and worst cases
    """
    # Setup plotting style
    setup_plot_style()

    # Get parameter names and save string
    param_names = get_param_names(N_LAYERS)
    case_mapping = ["Best Case"] * 5 + ["Median Case"] * 5 + ["Worst Case"] * 5

    # Convert to numpy array, correct physical ranges, and reorder parameters
    mc_data = val_unnormalized.clone().detach().cpu().numpy()
    mc_data[..., :6] = 10 ** mc_data[..., :6]  # mu_a, mu_s
    mc_data[..., -3:] = 10 ** mc_data[..., -3:]  #
    col_ids = np.array([i + j * N_LAYERS for i in range(N_LAYERS) for j in range(5)])
    mc_data = mc_data.reshape(-1, 15)[:, col_ids]

    # Define multi-header for Monte Carlo simulation
    dummy_wvl = np.array([400 * 10**-9])
    header = pd.MultiIndex.from_product(
        [
            [f"layer{i}" for i in range(N_LAYERS)],
            [np.round(dummy_wvl[0], 12)],
            ["ua", "us", "g", "n", "d"],
        ],
        names=["layer [top first]", "wavelength [m]", "parameter"],
    )

    def run_monte_carlo(
        params: np.ndarray, wavelengths: np.ndarray = dummy_wvl
    ) -> np.ndarray:
        """Run Monte Carlo simulation for given parameters and wavelengths."""
        mco_folder = ""
        ignore_a = True
        eps = 1e-8
        # NOTE: Needs correction for Shapley also using negative values,
        # thus clip to the correct range
        params[..., :6] = np.clip(params[..., :6], a_min=eps, a_max=None)
        params[..., 6:9] = np.clip(params[..., 6:9], a_min=-1 + eps, a_max=1 - eps)
        params[..., 9:12] = np.clip(params[..., 9:12], a_min=1 + eps, a_max=None)
        params[..., -3:] = np.clip(params[..., -3:], a_min=eps, a_max=None)

        # convert np array to DataFrame here (to avoid SHAP issues with DataFrame input)
        reflectances = calculate_spectrum_for_physical_batch(
            pd.DataFrame(params, columns=header),
            wavelengths,
            10**6,
            batch_id=str(0),
            mci_base_folder=mco_folder,
            ignore_a=ignore_a,
            mco_file="batch.mco",
        )

        return reflectances.reflectances.values

    # Select same best, median and worst case indices to make SHAP calculation bearable
    mc_cases = mc_data[case_indices]
    logger.info(f"Subset shape: {mc_cases.shape}")
    explainer = shap.Explainer(run_monte_carlo, mc_data[::SUBSAMPLE_FACTOR])
    expected_value = np.array([reflectance[::SUBSAMPLE_FACTOR].mean().cpu().numpy()])

    # Compute SHAP values takes between 1-3 minutes per sample, depending on GPU load
    shap_values_mc = np_cache_to_file(
        shapley_values_predict,
        "shap/mc_ground_truth_shap_values.npz",
    )(mc_cases, explainer, batch_size=1)
    logger.info(f"SHAP values shape: {shap_values_mc.shape}")

    # Generate summary plot
    shap.summary_plot(
        shap_values_mc,
        mc_cases,
        feature_names=param_names,
        show=False,
    )
    save_plot_notebook(f"{SAVE_PREFIX}summary_physical_mc_cases.png")

    for i, idx in track(enumerate(case_indices), description="Case Progress"):
        shap_waterfall_info = shap.Explanation(
            base_values=expected_value,
            values=shap_values_mc[i].squeeze(),
            feature_names=param_names,
            data=mc_cases[i],
        )
        shap.plots.waterfall(shap_waterfall_info, max_display=15, show=False)
        plt.tight_layout()
        plt.title(f"Waterfall Plot for {case_mapping[i]} {i % len(case_mapping)}")
        save_plot_notebook(
            f"{SAVE_PREFIX}waterfall_physical_mc_{case_mapping[i]}_{idx}.png"
        )

    return shap_values_mc


def plot_shap_values_comparison(
    shap_values_mlp: np.ndarray,
    shap_values_mc: np.ndarray,
    val_normalized: torch.Tensor,
    case_indices: np.ndarray,
) -> None:
    """Plot comparison of SHAP values between MLP and Monte Carlo.

    Args:
        shap_values_mlp: SHAP values for the MLP model
        shap_values_mc: SHAP values for the Monte Carlo model
        val_normalized: Normalized validation data
        case_indices: Indices of the best, median, and worst cases
    """
    # Setup plotting style
    setup_plot_style()

    # Get parameter names and save string
    param_names = get_param_names(N_LAYERS)

    # Plot difference between surrogate SHAP model and ground truth SHAP model
    shap_diff = shap_values_mc - shap_values_mlp.squeeze()

    shap.summary_plot(
        shap_diff,
        val_normalized.reshape(-1, 15)[case_indices],
        feature_names=param_names,
        show=False,
    )
    save_plot_notebook(f"{SAVE_PREFIX}summary_physical_mc_diff.png")

    # print difference MAE
    mae_per_sample = np.abs(shap_diff).mean(axis=1)
    mae_per_param = np.abs(shap_diff).mean(axis=0)

    bins = np.logspace(
        np.log10(mae_per_param.min() + 1e-3),
        np.log10(mae_per_param.max()),
        100,
    )
    plt.hist(mae_per_sample, bins=bins, label="Per Sample")
    plt.hist(mae_per_param, bins=bins, label="Per Parameter")
    plt.xlabel("Mean Absolute Error of SHAP Values")
    plt.xscale("log")
    plt.ylabel("Counts")
    plt.title("MAE of SHAP Values for Surrogate vs. Ground Truth Model")
    plt.legend()
    plt.tight_layout()
    save_plot_notebook(f"{SAVE_PREFIX}mae_shap_values_comparison.png")


def load_model_and_data() -> tuple[BaseModel, PreProcessor, torch.Tensor, torch.Tensor]:
    """Load the trained model and prepare validation data.

    Returns:
        Tuple of (model, preprocessor, validation_data, validation_normalized)

    Raises:
        FileNotFoundError: If model or data files are not found
        RuntimeError: If model loading or data processing fails
    """
    # Load configuration and model
    cfg = load_config()
    model, preprocessor, cfg = load_trained_model(
        os.path.join(
            os.environ["data_dir"], cfg["surrogate"]["issi_model"]["base_path"]
        ),
        cfg["surrogate"]["issi_model"]["checkpoint_path"],
        BaseModel,
    )

    # Load physiological data
    physiological_data = SimulationDataLoader().load_simulation_data(DATA_PATH)

    # Prepare validation dataset
    val = physiological_data[
        preprocessor.consistent_data_split_ids(physiological_data, mode="val")
    ]

    # Transform to normalized network parameters
    val_normalized = preprocessor(val)
    val_normalized = val_normalized[..., :-1]
    val_normalized[..., -1] = 0.0  # Set deepest layer to zero
    logger.info(f"Validation network input data shape: {val_normalized.shape}")

    return model, preprocessor, val, val_normalized


def compute_predictions(
    model: BaseModel,
    preprocessor: PreProcessor,
    val_normalized: torch.Tensor,
    val: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute model predictions and prepare data for analysis.

    Args:
        model: Trained model
        preprocessor: Data preprocessor for normalization
        val_normalized: Normalized validation data
        val: Original validation data

    Returns:
        Tuple of (reflectance, predicted_reflectance, val_unnormalized)
    """
    # Compute reflectances
    reflectance = val[..., -N_WAVELENGTHS:].unsqueeze(-1)
    predicted_reflectance = predict_in_batches(
        model,
        val_normalized,
        batch_size=BATCH_SIZE,
    ).reshape(len(val), N_WAVELENGTHS, 1)
    logger.info(
        f"Real and predicted reflectance shapes: {reflectance.shape}, "
        f"{predicted_reflectance.shape}"
    )

    # Undo normalization to logarithmic, "raw" physical parameters
    val_unnormalized = val_normalized * torch.maximum(
        preprocessor.norm_2, torch.tensor([1e-8])
    )
    val_unnormalized += preprocessor.norm_1
    logger.info(f"Validation network unnormalized data shape: {val_unnormalized.shape}")

    return reflectance, predicted_reflectance, val_unnormalized


def generate_surface_plots(
    val_unnormalized: torch.Tensor,
    reflectance: torch.Tensor,
    predicted_reflectance: torch.Tensor,
) -> None:
    """Generate 3D surface plots for visualization.

    Args:
        val_unnormalized: Unnormalized validation data
        reflectance: Ground truth reflectance values
        predicted_reflectance: Model predicted reflectance values
    """
    # Subsample data for plotting
    subsample_indices = slice(None, None, SUBSAMPLE_FACTOR)
    plot_data = val_unnormalized[subsample_indices].reshape(-1, N_WAVELENGTHS)

    # Plot 1: Ground truth reflectance
    plot_3d_surface(
        plot_data[:, 0],  # mu_a
        plot_data[:, 1],  # mu_s
        reflectance[subsample_indices].flatten(),
    )
    save_plot_notebook(f"{SAVE_PREFIX}surface_plot_ground_truth.png")

    # Plot 2: Prediction error
    plot_3d_surface(
        plot_data[:, 0],  # mu_a
        plot_data[:, 1],  # mu_s
        (
            reflectance[subsample_indices] - predicted_reflectance[subsample_indices]
        ).flatten(),
        symlog_z=True,
    )
    save_plot_notebook(f"{SAVE_PREFIX}surface_plot_prediction_error.png")


def main() -> None:
    """Main execution function for the explainability analysis."""
    logger.info("Step 1/7: Loading model and data...")
    model, preprocessor, val, val_normalized = load_model_and_data()
    logger.info("Step 2/7: Computing model predictions...")
    reflectance, predicted_reflectance, val_unnormalized = compute_predictions(
        model, preprocessor, val_normalized, val
    )
    logger.info("Step 3/7: Generating 3D surface plots...")
    generate_surface_plots(val_unnormalized, reflectance, predicted_reflectance)
    logger.info("Step 4/7: Generating surface and contour plots...")
    generate_surface_and_contour_plots(
        model,
        preprocessor,
        val_normalized,
        reflectance,
    )
    logger.info("Step 5/7: Analyzing model performance...")
    case_indices = analyze_model_performance(
        reflectance,
        predicted_reflectance,
    )
    logger.info("Step 6/7: Running SHAP analysis...")
    run_base_shap_analysis(model, val_normalized)
    logger.info(
        "Step 7/7: Running SHAP analysis for best, median, and worst cases "
        "in comparison to ground truth Monte Carlo simulation..."
    )
    shap_values_mlp = run_case_shap_analysis(model, val_normalized, case_indices)
    shap_values_mc = run_case_shap_analysis_monte_carlo(
        val_unnormalized, case_indices, reflectance
    )
    plot_shap_values_comparison(
        shap_values_mlp, shap_values_mc, val_normalized, case_indices
    )


if __name__ == "__main__":
    main()
