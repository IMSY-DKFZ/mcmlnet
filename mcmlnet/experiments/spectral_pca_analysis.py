"""
Compare reflectance distributions (PCA projections) of different simulation sets
with real data, aggregated at different levels (image, subject, organ).
Controlled by a YAML configuration file.
"""

import os
import pickle

import fastkde
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from dotenv import load_dotenv
from matplotlib.lines import Line2D
from matplotlib.path import Path
from rich.progress import track
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from sklearn.decomposition import IncrementalPCA
from torch.utils.data import DataLoader

from mcmlnet.experiments.data_loaders.aggregation import (
    aggregate_data_image_level,
    aggregate_data_subject_level,
)
from mcmlnet.experiments.data_loaders.config import (
    DataConfig,
)
from mcmlnet.experiments.data_loaders.real_data import (
    CombinedHumanDataLoader,
    PigDataLoader,
)
from mcmlnet.experiments.data_loaders.simulation import (
    SimulationDataLoaderManager,
)
from mcmlnet.experiments.data_loaders.utils import get_subject_id
from mcmlnet.experiments.plotting import (
    get_simulation_color,
    setup_plot_style,
)
from mcmlnet.utils.load_configs import (
    label_is_organic,
    load_config,
    load_label_map,
)

load_dotenv()
setup_plot_style()

NIR_ABLATION_CONFIG = {
    "enabled": False,
    "waveband_cutoff": 85,
    "suffix": "",
}
NIR_ABLATION_CONFIG["suffix"] = (
    "_NIR_ablation" if NIR_ABLATION_CONFIG["enabled"] else ""
)

if not os.path.exists(os.environ["plot_save_dir"]):
    os.makedirs(os.environ["plot_save_dir"])


def get_data_for_pca(
    config: dict[str, dict[str, str | bool | int | list[float]]],
    dataloader: DataLoader,
    label_map: dict[int, str],
) -> tuple[np.ndarray | None, pd.DataFrame | None]:
    """
    Collects the aggregated reflectance mean spectra and metadata
    for PCA fitting (or None for raw data).

    Args:
        config: Configuration dictionary.
        dataloader: DataLoader for real data.
        label_map: Dictionary mapping label IDs to names.

    Returns:
        Tuple containing:
            - Reflectance mean spectra (np.ndarray or None for raw data).
            - Metadata (pd.DataFrame or None for raw data).
    """
    agg_level = config["pca"]["aggregation_level"]
    dataset = config["data"]["real_data"]
    fit_organic = config["data"]["organic_only"]
    surrogate = config["pca"]["use_surrogate_model"]
    print(
        f"Preparing real data for PCA (aggregation: {agg_level}, "
        f"organic only: {fit_organic})..."
    )

    if agg_level in ["image", "subject"]:
        # Cache data aggregation
        cache_filename = os.path.join(
            os.environ["cache_dir"],
            f"mean_spectra_{dataset}_{surrogate}_{agg_level}_pca_components.pkl",
        )
        try:
            with open(cache_filename, "rb") as f:
                mean_spectra_dict = pickle.load(f)
            print(f"Loaded cached mean spectra for {agg_level} level.")
        except FileNotFoundError:
            print(f"Cache not found for {agg_level} level, aggregating data...")
            if agg_level == "image":
                mean_spectra_dict = aggregate_data_image_level(
                    dataloader,
                    dataset,  # type: ignore [arg-type]
                    fit_organic,  # type: ignore [arg-type]
                )
            elif agg_level == "subject":
                mean_spectra_dict = aggregate_data_subject_level(
                    dataloader,
                    dataset,  # type: ignore [arg-type]
                    fit_organic,  # type: ignore [arg-type]
                )
            else:
                raise ValueError(f"Unknown aggregation_level: {agg_level}") from None

            # Cache aggregated data
            with open(cache_filename, "wb") as f:
                pickle.dump(mean_spectra_dict, f)
            print(f"Cached mean spectra to: {cache_filename}")

    if agg_level == "image":
        reflectances = mean_spectra_dict["mean_spectra"]
        metadata = pd.DataFrame(
            {
                "label": mean_spectra_dict["label_ids"],
                "subject": mean_spectra_dict["subject_ids"],
                "image": mean_spectra_dict["image_ids"],
            }
        )
        metadata["label_name"] = metadata["label"].map(label_map)

        return reflectances, metadata

    elif agg_level == "subject":
        reflectances = mean_spectra_dict["mean_spectra"]
        metadata = pd.DataFrame(
            {
                "label": mean_spectra_dict["label_ids"],
                "subject": mean_spectra_dict["subject_ids"],
            }
        )
        metadata["label_name"] = metadata["label"].map(label_map)

        return reflectances, metadata

    elif agg_level == "raw":
        # For 'raw' level, (I)PCA is fitted iteratively on raw pixels
        # in the fit_pca function.
        return None, None

    else:
        raise ValueError(f"Unknown aggregation_level: {agg_level}")


def fit_pca(
    config: dict[str, dict[str, str | bool | int | list[float]]],
    reflectances: np.ndarray | None,
    dataloader: DataLoader,
    n_wvl: int,
) -> IncrementalPCA:
    """
    Fits IncrementalPCA or loads from cache.

    Args:
        config: Experiment configuration dictionary.
        reflectances: Reflectances for PCA fitting (None if 'raw' level).
        dataloader: DataLoader for raw data (only needed for 'raw' level).
        n_wvl: Number of wavelengths.

    Returns
        Fitted IncrementalPCA model.
    """
    dataset = config["data"]["real_data"]
    test_data = config["data"]["test_data"]
    specular = config["data"]["specular"]
    agg_level = config["pca"]["aggregation_level"]
    batch_size = config["pca"]["batch_size"]
    minimal_overlap = config["pca"]["minimal_overlap"]
    n_components = n_wvl

    # Construct cache filename - only used for raw data PCA (aggregation level 'raw')
    cache_filename = os.path.join(
        os.environ["cache_dir"],
        f"ipca_{dataset}_{test_data}_{specular}_{minimal_overlap}_{n_components}comp.pkl",
    )

    # Check if we need to fit PCA or load from cache
    try:
        with open(cache_filename, "rb") as f:
            ipca = pickle.load(f)
    except FileNotFoundError:
        ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

        if agg_level == "raw":
            for _i, batch in track(
                enumerate(dataloader),
                total=len(dataloader),
                description="PCA Partial Fit",
            ):
                # load and clamp spectra to physically meaningful values
                img = torch.clamp(batch["features"], 0, 1)
                img = img.view(-1, img.shape[-1]).cpu().numpy()
                # fit only on the wavelengths present in the reference simulation
                ipca.partial_fit(img[:, :n_wvl])
            print(f"Caching PCA model to: {cache_filename}")
            with open(cache_filename, "wb") as f:
                pickle.dump(ipca, f)
        elif reflectances is not None:
            ipca.fit(reflectances[:, :n_wvl])
        else:
            raise RuntimeError(
                "Cannot fit PCA: No data matrix provided "
                "and aggregation level is not 'raw'."
            ) from None

    return ipca


def transform_data(
    ipca: IncrementalPCA,
    n_comp_plot: int,
    n_wvl: int,
    config: dict[str, dict[str, str | bool | int | list[float]]],
    reflectances: np.ndarray | None,
    metadata: pd.DataFrame | None,
    dataloader: DataLoader | None,
    label_map: dict[int, str] | None,
) -> pd.DataFrame:
    """Transforms datasets using the fitted PCA model."""
    dataset = config["data"]["real_data"]
    agg_level = config["pca"]["aggregation_level"]
    n_comp_plot = min(n_comp_plot, ipca.n_components_)

    if agg_level == "raw" and reflectances is None:
        all_transformed_real = []
        all_labels = []
        all_subjects = []
        # Iterative transformation of raw pixels
        for _i, batch in track(
            enumerate(dataloader),  # type: ignore [arg-type]
            total=len(dataloader),  # type: ignore [arg-type]
            description="Transforming Real Data",
        ):
            # Load and clamp spectra to physically meaningful values
            img = torch.clamp(batch["features"], 0, 1)
            img = img.view(-1, img.shape[-1]).cpu().numpy()
            # Transform and collect spectra
            transformed_batch = ipca.transform(img[:, :n_wvl])[:, :n_comp_plot]
            all_transformed_real.append(transformed_batch)
            # Collect metadata
            batch_labels = batch["labels"].view(-1).cpu().numpy()
            all_labels.append(batch_labels)
            subject_id = get_subject_id(batch["image_name"][0], dataset)  # type: ignore [arg-type]
            all_subjects.append([subject_id] * len(batch_labels))

        # Concatenate all transformed data
        transformed_np = np.concatenate(all_transformed_real, axis=0)
        transformed_df = pd.DataFrame(
            transformed_np, columns=[f"PC_{i}" for i in range(n_comp_plot)]
        )
        transformed_df["label"] = np.concatenate(all_labels, axis=0)
        transformed_df["subject"] = np.concatenate(all_subjects, axis=0)
        transformed_df["label_name"] = transformed_df["label"].map(label_map)

    else:
        # Transform pre-aggregated data/ non-iterative dataset
        transformed_np = ipca.transform(reflectances[:, :n_wvl])[:, :n_comp_plot]  # type: ignore [index]
        transformed_df = pd.DataFrame(
            transformed_np, columns=[f"PC_{i}" for i in range(n_comp_plot)]
        )
        if metadata is not None:
            # Add metadata to transformed data
            transformed_df["label"] = metadata["label"].values
            transformed_df["subject"] = metadata["subject"].values
            transformed_df["label_name"] = metadata["label_name"].values

    return transformed_df


def plot_pca_variance_and_components(
    pca: IncrementalPCA,
    save_dir: str,
    prefix: str,
) -> None:
    """Plot the PCA components and explained variance."""
    variance_path = os.path.join(save_dir, f"{prefix}_pca_explained_variance.png")
    animation_path = os.path.join(save_dir, f"{prefix}_pca_components_animation.gif")
    last_components_path = os.path.join(save_dir, f"{prefix}_pca_components_last.png")

    # Plot the explained variance
    plt.figure(figsize=(12, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.title("Cumulative Explained Variance")
    plt.xlabel("Number of Components")
    plt.ylabel("Explained Variance")
    plt.grid()
    plt.tight_layout()
    plt.savefig(variance_path)
    plt.close()
    print(f"Saved PCA variance plot to {variance_path}")

    # Animate all PCA components
    fig, ax = plt.subplots(figsize=(12, 5))

    def update(frame: int) -> None:
        ax.clear()
        ax.plot(pca.components_.T[:, frame], label=f"Component {frame}")
        ax.set_title("PCA Components Animation")
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Component Value")
        ax.grid()
        ax.legend()
        plt.tight_layout()

    # Save the animation as a GIF
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=range(pca.components_.shape[0]),
        interval=1000,
        repeat=True,
    )
    ani.save(animation_path)
    print(f"Saved PCA components animation to {animation_path}")
    plt.close(fig)

    # Plot last 10 PCA components
    plt.figure(figsize=(12, 5))
    num_components_to_plot = min(10, pca.components_.shape[0])
    plt.plot(pca.components_.T[:, -num_components_to_plot:])
    plt.title(f"Last {num_components_to_plot} PCA Components")
    plt.xlabel("Wavelength")
    plt.ylabel("Component Value")
    plt.grid()
    plt.tight_layout()
    plt.savefig(last_components_path)
    plt.close()
    print(f"Saved last PCA components plot to {last_components_path}")


def matplotlit_jointplot(
    x: np.ndarray,
    y: np.ndarray,
    fig: plt.Figure,
    grid: plt.GridSpec,
    color: str | tuple,
    cmap: str | None,
    gridsize: int = 200,
    mincnt: float = 0.1,
    maxcnt_scale: float = 0.5,
    axes: tuple[plt.Axes, plt.Axes, plt.Axes] | None = None,
    kind: str = "hist2d",
    bin_scale: str = "log",
    bin_alpha: float = 1.0,
) -> tuple[plt.Axes, plt.Axes, plt.Axes]:
    """Create a jointplot with hexbin/hist2d and marginal histograms."""
    # Unpack or define subplots
    if axes is not None:
        main_ax, x_hist_ax, y_hist_ax = axes
    else:
        main_ax = fig.add_subplot(grid[1:, :-1])
        x_hist_ax = fig.add_subplot(grid[0, :-1], sharex=main_ax)
        y_hist_ax = fig.add_subplot(grid[1:, -1], sharey=main_ax)

    # Set color for the main plot
    if isinstance(color, tuple):
        # Convert RGB tuple (0-1 range) to hex if needed
        try:
            plot_color = mcolors.to_hex(color)
        except ValueError:
            # Assume it's already a valid color string or handle error
            plot_color = color
    else:
        plot_color = color

    # Ensure x and y are 1D arrays
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    # Define main plot
    if kind == "hex":
        main_ax.hexbin(
            x,
            y,
            gridsize=gridsize,
            cmap=cmap,
            mincnt=mincnt,
            bins=bin_scale,
            alpha=bin_alpha,
        )
    elif kind == "hist2d":
        _2d_hist = np.histogram2d(x, y, bins=gridsize)[0]
        # Adapt color map to increase the contrast of low counts
        norm = mcolors.LogNorm(
            vmin=max(1e-1, mincnt), vmax=np.max(_2d_hist) * maxcnt_scale
        )
        main_ax.hist2d(
            x,
            y,
            bins=gridsize,
            cmap=cmap,
            alpha=bin_alpha,
            norm=norm,
            cmin=mincnt,
        )
    elif kind == "kde":
        try:
            sns.kdeplot(
                x=x,
                y=y,
                ax=main_ax,
                color="black",
                fill=False,
                bw_adjust=0.5,
                thresh=0.005,
                levels=5,
            )
        except Exception as e:
            print(f"Error in kdeplot: {e}")
            print("Check if the data is suitable for KDE.")
    elif kind == "fastkde":
        try:
            pdf = fastkde.pdf(x, y, var_names=["x", "y"], num_points_per_sigma=10)
            # Apply (empirical) threshold the KDE
            z = pdf.to_numpy()
            z[z < 0.005] = 0
            # Plot fastKDE contours
            main_ax.contour(
                pdf.x,
                pdf.y,
                z,
                levels=5,
                linewidths=1,
                colors=plot_color,
                alpha=0.7,
            )
        except Exception as e:
            print(f"Error in kdeplot: {e}")
            print("Check if the data is suitable for KDE.")
    elif kind == "hist2d+kde":
        _2d_hist = np.histogram2d(x, y, bins=gridsize)[0]
        # Adapt color map to increase the contrast of low counts
        norm = mcolors.LogNorm(
            vmin=max(1e-1, mincnt), vmax=np.max(_2d_hist) * maxcnt_scale
        )
        main_ax.hist2d(
            x, y, bins=gridsize, cmap=cmap, alpha=bin_alpha, norm=norm, cmin=mincnt
        )
        try:
            pdf = fastkde.pdf(x, y, var_names=["x", "y"], num_points_per_sigma=10)
            # Apply (empirical) threshold the KDE
            z = pdf.to_numpy()
            z[z < 0.005] = 0
            # Plot fastKDE contours
            main_ax.contour(
                pdf.x,
                pdf.y,
                z,
                levels=5,
                colors=plot_color,
                alpha=0.7,
            )
        except Exception as e:
            print(f"Error in kdeplot: {e}")
            print("Check if the data is suitable for KDE.")
    elif kind == "marginal_hist_only":
        pass
    else:
        raise NotImplementedError(f"Kind {kind} not implemented")

    # Define marginal histograms
    x_hist_ax.hist(
        x,
        bins=gridsize,
        density=True,
        color=plot_color,
        edgecolor=plot_color,
        alpha=bin_alpha,
    )
    y_hist_ax.hist(
        y,
        bins=gridsize,
        density=True,
        color=plot_color,
        edgecolor=plot_color,
        alpha=bin_alpha,
        orientation="horizontal",
    )
    # Hide axis labels for cleaner visualization
    x_hist_ax.axis("off")
    y_hist_ax.axis("off")
    main_ax.grid(True)

    # Remove tick labels on shared axes
    plt.setp(x_hist_ax.get_xticklabels(), visible=False)
    plt.setp(y_hist_ax.get_yticklabels(), visible=False)

    return main_ax, x_hist_ax, y_hist_ax


def plot_aggregated_data(
    ipca: IncrementalPCA,
    transformed_real: pd.DataFrame,
    transformed_sim: pd.DataFrame,
    transformed_related: pd.DataFrame,
    data_key: str,
    config: dict[str, dict[str, str | bool | int | list[float]]],
    label: int | None = None,
    scatter_color_map: dict[str, str] | str | None = None,
    scatter_size: int = 1,
    scatter_alpha: float = 1.0,
) -> None:
    """Generates plots for the 'image' and 'subject' aggregation level."""
    dataset = config["data"]["real_data"]
    agg_level = config["pca"]["aggregation_level"]
    n_comp_plot = config["pca"]["n_components_plot"]
    specular_suffix = "_specular" if config["data"]["specular"] else ""
    overlap_suffix = "_minimal_overlap" if config["pca"]["minimal_overlap"] else ""
    surrogate_suffix = "_surrogate" if config["pca"]["use_surrogate_model"] else ""
    plot_inverse = config["pca"]["show_inverse"]

    # Define colors for real data labels
    unique_labels = transformed_real["label_name"].unique()
    # https://stackoverflow.com/questions/55469432/is-there-a-similar-color-palette-to-tab20c-with-bigger-number-of-colors
    cmap_real = mcolors.ListedColormap(
        plt.get_cmap("gist_rainbow")(np.linspace(0, 1, len(unique_labels)))
    )
    if scatter_color_map is None:
        scatter_color_map = dict(zip(unique_labels, cmap_real.colors, strict=False))
    elif scatter_color_map == "black":
        scatter_color_map = dict.fromkeys(unique_labels, "black")

    # Iterate over the PC pairs
    for pc_pair_start in range(n_comp_plot // 2):  # type: ignore [operator]
        pc_x_idx = pc_pair_start * 2
        pc_y_idx = 1 + pc_pair_start * 2
        pc_x = f"PC_{pc_x_idx}"
        pc_y = f"PC_{pc_y_idx}"

        # Adjust grid spec if not plotting inverse transforms
        fig = (
            plt.figure(figsize=(20, 10))
            if plot_inverse
            else plt.figure(figsize=(3.2, 2.8))
        )
        grid_cols = 12 if (plot_inverse and pc_pair_start == 0) else 6
        grid_rows = 6
        grid = plt.GridSpec(grid_rows, grid_cols, fig)

        # Set axis ranges
        x_min = np.min(
            np.concatenate(
                [
                    transformed_real[pc_x],
                    transformed_sim[pc_x],
                    transformed_related[pc_x],
                ]
            )
        )
        x_max = np.max(
            np.concatenate(
                [
                    transformed_real[pc_x],
                    transformed_sim[pc_x],
                    transformed_related[pc_x],
                ]
            )
        )
        y_min = np.min(
            np.concatenate(
                [
                    transformed_real[pc_y],
                    transformed_sim[pc_y],
                    transformed_related[pc_y],
                ]
            )
        )
        y_max = np.max(
            np.concatenate(
                [
                    transformed_real[pc_y],
                    transformed_sim[pc_y],
                    transformed_related[pc_y],
                ]
            )
        )
        x_padding = (x_max - x_min) * 0.05
        y_padding = (y_max - y_min) * 0.05
        x_lim = (x_min - x_padding, x_max + x_padding)
        y_lim = (y_min - y_padding, y_max + y_padding)

        # Define PCA plot main and marginal histogram axes
        main_ax = fig.add_subplot(grid[1:, :5])
        x_hist_ax = fig.add_subplot(grid[0, :5], sharex=main_ax)
        y_hist_ax = fig.add_subplot(grid[1:, 5], sharey=main_ax)
        plt.setp(x_hist_ax.get_xticklabels(), visible=False)
        plt.setp(y_hist_ax.get_yticklabels(), visible=False)

        # Add marginal histograms
        main_ax, x_hist_ax, y_hist_ax = matplotlit_jointplot(
            transformed_real[pc_x],
            transformed_real[pc_y],
            fig,
            grid,
            axes=(main_ax, x_hist_ax, y_hist_ax),
            color="black",
            cmap=None,
            kind="marginal_hist_only",
        )

        # Plot simulated data as reference(s)
        cmap1 = mcolors.LinearSegmentedColormap.from_list(
            "custom_transformed", ["white", get_simulation_color("generic_sims")]
        )
        main_ax, x_hist_ax, y_hist_ax = matplotlit_jointplot(
            transformed_sim[pc_x],
            transformed_sim[pc_y],
            fig,
            grid,
            axes=(main_ax, x_hist_ax, y_hist_ax),
            color=get_simulation_color("generic_sims"),
            cmap=cmap1,
            kind="fastkde",
            bin_alpha=0.2,
        )
        cmap2 = mcolors.LinearSegmentedColormap.from_list(
            "custom_transformed", ["white", get_simulation_color(data_key)]
        )
        main_ax, x_hist_ax, y_hist_ax = matplotlit_jointplot(
            transformed_related[pc_x],
            transformed_related[pc_y],
            fig,
            grid,
            axes=(main_ax, x_hist_ax, y_hist_ax),
            color=get_simulation_color(data_key),
            cmap=cmap2,
            kind="fastkde",
            bin_alpha=0.2,
        )

        # Plot real data
        if config["pca"]["aggregation_level"] == "raw":
            main_ax, x_hist_ax, y_hist_ax = matplotlit_jointplot(
                transformed_real[pc_x],
                transformed_real[pc_y],
                fig,
                grid,
                axes=(main_ax, x_hist_ax, y_hist_ax),
                color="black",
                cmap="Greys",
                kind="fastkde",
                bin_alpha=0.5,
            )
        else:
            # define marker and color mapping manually
            marker_list = [
                ".",
                "v",
                "^",
                "<",
                ">",
                # "1",
                # "2",
                # "3",
                # "4",
                "8",
                "s",
                "p",
                "P",
                "*",
                "h",
                "H",
                # "+",
                # "x",
                "X",
                "D",
                "d",
                # "|",
                # "_",
            ]
            label_to_marker = {
                label: marker_list[i % len(marker_list)]
                for i, label in enumerate(unique_labels)
            }
            if isinstance(scatter_color_map, str):
                cmap = plt.get_cmap(scatter_color_map)
                scatter_color_map = {
                    label: cmap(i % 20) for i, label in enumerate(unique_labels)
                }

            # Scatter color-coded, aggregated data
            for _label, _group in transformed_real.groupby("label_name"):
                main_ax.scatter(
                    _group[pc_x],
                    _group[pc_y],
                    color=scatter_color_map[_label],
                    marker=label_to_marker[_label],
                    s=scatter_size,
                    alpha=scatter_alpha,
                    edgecolor="black",
                    linewidth=0.1,
                )
            # Calculate and plot the average (2D) value for each class
            averages = (
                transformed_real.groupby("label_name")[[pc_x, pc_y]]
                .mean()
                .reset_index()
            )
            for _, row in averages.iterrows():
                main_ax.scatter(
                    row[pc_x],
                    row[pc_y],
                    color=scatter_color_map[row["label_name"]],
                    s=25 * scatter_size,
                    edgecolor="black",
                    linewidth=0.5,
                    marker=label_to_marker[row["label_name"]],
                    zorder=10,
                )
        # Inverse Transform Examples (Optional, only for first PC pair)
        spectra_axes = None
        if plot_inverse and pc_pair_start == 0:
            n_test_pts_side = 6
            spectra_axes = np.zeros((n_test_pts_side, n_test_pts_side), dtype=object)
            for _x in range(n_test_pts_side):
                for _y in range(n_test_pts_side):
                    spectra_axes[_x, _y] = fig.add_subplot(
                        grid[_x, n_test_pts_side + _y]
                    )

            corners = np.array(
                [
                    [x_lim[0], y_lim[1]],  # top left
                    [x_lim[0], y_lim[0]],  # bottom left
                    [x_lim[1], y_lim[1]],  # top right
                    [x_lim[1], y_lim[0]],  # bottom right
                ]
            )
            test_pts_pc = np.array(
                [
                    (corners[0] * (1 - x) + corners[1] * x) * (1 - y)
                    + (corners[2] * (1 - x) + corners[3] * x) * y
                    for x in np.linspace(0.1, 0.9, n_test_pts_side)  # Avoid exact edges
                    for y in np.linspace(0.1, 0.9, n_test_pts_side)
                ]
            )

            # Create full PC space points (pad with zeros for other components)
            test_pts_full = np.zeros((test_pts_pc.shape[0], ipca.n_components_))
            test_pts_full[:, pc_x_idx] = test_pts_pc[:, 0]
            test_pts_full[:, pc_y_idx] = test_pts_pc[:, 1]
            inv_transformed_points = ipca.inverse_transform(test_pts_full)

            # Plot markers corresponding to inverse spectra on main plot
            main_ax.scatter(
                test_pts_pc[:, 0],
                test_pts_pc[:, 1],
                marker="x",
                c="k",
                s=20,
            )
            # Plot inverse transformed spectra
            for idx, (_x, _y) in enumerate(np.ndindex(spectra_axes.shape)):
                ax = spectra_axes[_x, _y]
                ax.plot(inv_transformed_points[idx])
                ax.set(xticks=[], yticks=[], ylim=(0, 1))
                ax.tick_params(axis="both", which="both", length=0)

        # Update axes limits
        main_ax.set_xlim(x_lim)
        main_ax.set_ylim(y_lim)
        x_hist_ax.set_xlim(x_lim)
        y_hist_ax.set_ylim(y_lim)

        # Customize legend (update alpha and size, add simulation data handles)
        leg_labels = transformed_real["label_name"].unique().tolist()
        custom_handles = []
        for _label in leg_labels:
            color = scatter_color_map.get(_label, "black")  # type: ignore [union-attr]
            custom_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=label_to_marker[_label],
                    color="w",
                    markerfacecolor=color,
                    markersize=5,
                    alpha=1.0,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                )
            )
        labels = [
            DataConfig.SIMULATION_NAMES["generic_sims"],
            DataConfig.SIMULATION_NAMES[data_key],
        ]
        sim_patch = mpatches.Patch(
            color=get_simulation_color("generic_sims"),
            alpha=0.6,
            label=DataConfig.SIMULATION_NAMES["generic_sims"],
        )
        related_patch = mpatches.Patch(
            color=get_simulation_color(data_key),
            alpha=0.6,
            label=DataConfig.SIMULATION_NAMES[data_key],
        )
        if config["pca"]["aggregation_level"] == "raw":
            real_patch = mpatches.Patch(
                color="black",
                alpha=0.6,
                label="Human Data",
            )
            labels += ["Human Data"]
        else:
            real_patch = mpatches.Patch(
                color="black",
                alpha=0.6,
                label="Human Data (Marginals)",
            )
            labels += ["Human Data (Marginals)"]
        all_handles = [sim_patch, related_patch, real_patch, *custom_handles]
        main_ax.legend(
            handles=all_handles,
            labels=labels + leg_labels,
            title="Data Source / Label",
            ncols=1,
            loc="lower center",
            bbox_to_anchor=(
                0.5,
                1.05,
            ),
            ncol=4,
        )
        main_ax.set_xlabel(
            f"{pc_x.replace('_', ' ')} (Explained Variance "
            f"{ipca.explained_variance_ratio_[pc_x_idx] * 100:.2f}%)"
        )
        main_ax.set_ylabel(
            f"{pc_y.replace('_', ' ')} (Explained Variance "
            f"{ipca.explained_variance_ratio_[pc_y_idx] * 100:.2f}%)"
        )
        main_ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        # Write to file
        save_filename = (
            f"{dataset}_pca_{pc_x}-{pc_y}_{agg_level}_vs_{data_key}"
            f"{surrogate_suffix}{overlap_suffix}{specular_suffix}{NIR_ABLATION_CONFIG['suffix']}"
        )
        if label is not None:
            save_filename += f"_label_{label}"
        save_filename += ".svg"
        save_path = os.path.join(os.environ["plot_save_dir"], save_filename)
        print(f"Saving plot to: {save_path}")
        plt.savefig(save_path, dpi=600)
        plt.close(fig)


def plot_raw_organ_level(
    ipca: IncrementalPCA,
    transformed_real: pd.DataFrame,
    transformed_sim: pd.DataFrame,
    transformed_related: pd.DataFrame,
    data_key: str,
    config: dict[str, dict[str, str | bool | int | list[float]]],
    label_map: dict[int, str],
) -> None:
    """Generates plots for the 'raw' PCA plot (per organ)."""
    # Wrap around aggregated data function and filter data by label
    unique_labels = transformed_real["label"].unique()
    if config["data"]["organic_only"]:
        organic_labels = label_is_organic(unique_labels, config["data"]["real_data"])  # type: ignore [arg-type]
        unique_labels = unique_labels[organic_labels]

    for label in unique_labels:
        _transformed_real = transformed_real[transformed_real["label"] == label].copy()
        plot_aggregated_data(
            ipca,
            _transformed_real,
            transformed_sim,
            transformed_related,
            data_key,
            config,
            label=label_map[label],  # type: ignore [arg-type]
            scatter_color_map="black",
            scatter_size=1,
            scatter_alpha=0.5,
        )


def plot_outliers_with_nearest_neighbour_convex_hull(
    transformed_real: pd.DataFrame,
    transformed_sim: pd.DataFrame,
    real_spectra: np.ndarray,
    sim_spectra: np.ndarray,
    pc_x: str,
    pc_y: str,
    prefix: str,
    n_examples: int = 10,
) -> None:
    """
    Plot example real spectra outliers with their nearest neighbour
    in the simulation data.
    """
    # Get two principal components
    sim_points = transformed_sim[[pc_x, pc_y]].to_numpy()
    real_points = transformed_real[[pc_x, pc_y]].to_numpy()

    # Compute convex hull of simulation
    hull = ConvexHull(sim_points)
    hull_path = Path(sim_points[hull.vertices])

    # Find real points outside the hull
    is_outside = ~hull_path.contains_points(real_points)
    outlier_idx = np.where(is_outside)[0]

    # Subsample outliers if too many
    if len(outlier_idx) == 0:
        print(f"No real points outside simulation convex hull (prefix: {prefix}).")
        return
    if len(outlier_idx) > n_examples:
        np.random.seed(42)
        outlier_idx = np.random.choice(outlier_idx, n_examples, replace=False)

    plt.figure(figsize=(12, 8))
    for idx in outlier_idx:
        # Find nearest sim point for each outlier
        real = real_spectra[idx]
        argmin_idx = np.argmin(cdist(np.expand_dims(real, 0), sim_spectra))
        sim = sim_spectra[argmin_idx]

        # Plot real spectra as well as the nearest neighbour simulation spectrum
        plt.plot(real, "tab:blue", alpha=0.8, label=f"Real Outlier {real_points[idx]}")
        plt.plot(
            sim, "tab:orange", alpha=0.8, label=f"Nearest Sim {sim_points[argmin_idx]}"
        )

    plt.xlabel("Wavelength")
    plt.ylabel("Reflectance")
    plt.title("Outlier Real Spectra with Nearest Neighbour Simulation")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    save_filename = (
        f"{prefix}_outliers_nearest_neighbour_{pc_x}_{pc_y}"
        f"{NIR_ABLATION_CONFIG['suffix']}.png"
    )
    plt.savefig(os.path.join(os.environ["plot_save_dir"], save_filename), dpi=600)
    plt.close()

    # Plot the convex hull
    hull_vertices = np.append(hull.vertices, hull.vertices[0])
    plt.figure(figsize=(12, 8))
    plt.plot(
        sim_points[hull_vertices, 0],
        sim_points[hull_vertices, 1],
        "r--",
        lw=2,
        label="Convex Hull",
    )
    plt.hist2d(
        sim_points[:, 0],
        sim_points[:, 1],
        bins=100,
        cmap="Oranges",
        alpha=0.5,
        label="Simulation Density",
    )
    plt.scatter(
        real_points[outlier_idx, 0],
        real_points[outlier_idx, 1],
        c="tab:blue",
        s=10,
        marker="x",
        alpha=1.0,
        label="Real Outliers",
    )
    plt.xlabel(pc_x)
    plt.ylabel(pc_y)
    plt.title("Convex Hull of Simulation Data with Real Outliers")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    save_filename = (
        f"{prefix}_convex_hull_{pc_x}_{pc_y}{NIR_ABLATION_CONFIG['suffix']}.png"
    )
    plt.savefig(os.path.join(os.environ["plot_save_dir"], save_filename), dpi=600)
    plt.close()


def plot_outliers_with_nearest_neighbour(
    transformed_real: pd.DataFrame,
    transformed_sim: pd.DataFrame,
    real_spectra: np.ndarray,
    sim_spectra: np.ndarray,
    pc_x: str,
    pc_y: str,
    prefix: str,
    n_examples: int = 10,
) -> None:
    """
    Plot example real spectra outliers with their nearest neighbour
    in the simulation data.
    """
    # Get two principal components
    sim_points = transformed_sim[[pc_x, pc_y]].to_numpy()
    real_points = transformed_real[[pc_x, pc_y]].to_numpy()

    # Get outliers via 2D histogram edges
    bins = 200
    hist, xedges, yedges = np.histogram2d(sim_points[:, 0], sim_points[:, 1], bins=bins)

    # Get bin index for each real point
    x_bin_idx = np.searchsorted(xedges, real_points[:, 0], side="right") - 1
    y_bin_idx = np.searchsorted(yedges, real_points[:, 1], side="right") - 1

    # Mask points that fall outside the histogram range
    valid = (
        (x_bin_idx >= 0) & (x_bin_idx < bins) & (y_bin_idx >= 0) & (y_bin_idx < bins)
    )
    # Get the counts for each real point
    counts = np.zeros(real_points.shape[0], dtype=int)
    counts[valid] = hist[x_bin_idx[valid], y_bin_idx[valid]]
    # Identify outliers as previously empty bins, to which now a value is assigned
    outlier_idx = np.where(counts < 1)[0]

    # Subsample outliers if too many
    if len(outlier_idx) == 0:
        print(f"No outliers found (prefix: {prefix}).")
        return
    if len(outlier_idx) > n_examples:
        np.random.seed(42)
        outlier_idx = np.random.choice(outlier_idx, n_examples, replace=False)

    _fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    # Plot spectra in first subplot
    for idx in outlier_idx:
        # Find nearest sim point for each outlier
        real = real_spectra[idx]
        argmin_idx = np.argmin(cdist(np.expand_dims(real, 0), sim_spectra))
        sim = sim_spectra[argmin_idx]

        # Plot real spectra as well as the nearest neighbour simulation spectrum
        axs[0].plot(
            real, "tab:blue", alpha=0.8, label=f"Real Outlier {real_points[idx]}"
        )
        axs[0].plot(
            sim, "tab:orange", alpha=0.8, label=f"Nearest Sim {sim_points[argmin_idx]}"
        )

    axs[0].set_xlabel("Wavelength")
    axs[0].set_ylabel("Reflectance")
    axs[0].set_title("Outlier Real Spectra with Nearest Neighbour Simulation")
    axs[0].legend()
    axs[0].grid()
    # Plot the 2D histogram and outliers in PC space
    axs[1].hist2d(
        sim_points[:, 0],
        sim_points[:, 1],
        bins=bins,
        cmap="Oranges",
        alpha=0.5,
        label="Simulation Density",
    )
    axs[1].scatter(
        real_points[outlier_idx, 0],
        real_points[outlier_idx, 1],
        c="black",
        s=10,
        marker="x",
        label="Real Outliers",
    )
    axs[1].set_xlabel(pc_x)
    axs[1].set_ylabel(pc_y)
    axs[1].set_title("Convex Hull of Simulation Data with Real Outliers")
    axs[1].legend()
    axs[1].grid()
    plt.tight_layout()
    save_filename = (
        f"{prefix}_outliers_{pc_x}_{pc_y}{NIR_ABLATION_CONFIG['suffix']}.png"
    )
    plt.savefig(os.path.join(os.environ["plot_save_dir"], save_filename), dpi=600)
    plt.close()


def main() -> None:
    """Main function to execute the PCA analysis pipeline.

    Steps:
    1. Loads configuration and label maps.
    2. Loads real-world spectral data and simulation datasets.
    3. For each simulation dataset:
        a. Fits a PCA model.
        b. Transforms the simulated data.
        c. Plots the PCA variance and components.
        d. Plots the PCA transformed data.
        e. Plots example outliers with nearest neighbour.
    """
    # Load config and label map
    cfg = load_config()
    label_map = load_label_map(dataset=cfg["data"]["real_data"])

    # Get dataloader
    data_loader: DataLoader
    if cfg["data"]["real_data"] == "human":
        data_loader = CombinedHumanDataLoader()
    else:
        data_loader = PigDataLoader()
    if cfg["data"]["test_data"] == "test":
        dataloader = data_loader.load_test_data()
    elif cfg["data"]["test_data"] == "final":
        dataloader = data_loader.load_final_data()
    else:
        dataloader, _ = data_loader.load_training_data()

    # Load adapted simulation datasets
    if cfg["pca"]["use_surrogate_model"] is True:
        simulation_data = SimulationDataLoaderManager(
            cfg["data"]["specular"]
        ).load_surrogate_model_data()
    else:
        simulation_data = SimulationDataLoaderManager(
            cfg["data"]["specular"]
        ).load_simulation_data()

    # Collect the aggregated data for PCA fitting
    reflectances, metadata = get_data_for_pca(
        cfg,
        dataloader,
        label_map,
    )

    for data_key, data in simulation_data.items():
        # Skip generic simulations (always used as reference)
        if data_key == "generic_sims":
            continue
        # Ablation: NIR cropping
        if NIR_ABLATION_CONFIG["enabled"]:
            data = data[:, : NIR_ABLATION_CONFIG["waveband_cutoff"]]  # type: ignore [misc]
        # Choose to only use the global intersection of all simulation data
        if cfg["pca"]["minimal_overlap"] is True:
            n_wvl = np.min([_data.shape[1] for _data in simulation_data.values()])
        else:
            n_wvl = data.shape[1]

        # Fit PCA on the real (aggregated) data
        ipca = fit_pca(cfg, reflectances, dataloader, n_wvl)

        # Plot PCA variance and components
        specular_suffix = "_specular" if cfg["data"]["specular"] else ""
        overlap_suffix = "_minimal_overlap" if cfg["pca"]["minimal_overlap"] else ""
        surrogate_suffix = "_surrogate" if cfg["pca"]["use_surrogate_model"] else ""
        save_filename = (
            f"{cfg['data']['real_data']}_{data_key}"
            f"{surrogate_suffix}_{cfg['pca']['aggregation_level']}"
            f"{NIR_ABLATION_CONFIG['suffix']}{overlap_suffix}{specular_suffix}"
        )
        plot_pca_variance_and_components(
            ipca,
            os.environ["plot_save_dir"],
            prefix=save_filename,
        )

        # Transform the simulated data
        transformed_sim = transform_data(
            ipca,
            n_comp_plot=cfg["pca"]["n_components_plot"],
            n_wvl=n_wvl,
            config=cfg,
            reflectances=simulation_data["generic_sims"],
            metadata=None,
            dataloader=None,
            label_map=None,
        )
        transformed_related = transform_data(
            ipca,
            n_comp_plot=cfg["pca"]["n_components_plot"],
            n_wvl=n_wvl,
            config=cfg,
            reflectances=data,
            metadata=None,
            dataloader=None,
            label_map=None,
        )
        # transform the real data
        transformed_real = transform_data(
            ipca,
            n_comp_plot=cfg["pca"]["n_components_plot"],
            n_wvl=n_wvl,
            config=cfg,
            reflectances=reflectances,
            metadata=metadata,
            dataloader=dataloader,
            label_map=label_map,
        )

        # Plot
        agg_level = cfg["pca"]["aggregation_level"]
        if agg_level == "image" or agg_level == "subject":
            plot_aggregated_data(
                ipca,
                transformed_real,
                transformed_sim,
                transformed_related,
                data_key,
                cfg,
                scatter_color_map="tab20",
                scatter_alpha=0.5,
            )
        else:
            plot_raw_organ_level(
                ipca,
                transformed_real,
                transformed_sim,
                transformed_related,
                data_key,
                cfg,
                label_map=label_map,
            )

        if agg_level != "raw":
            # Plot example outliers with nearest neighbour
            plot_outliers_with_nearest_neighbour(
                transformed_real,
                transformed_sim,
                real_spectra=reflectances[:, :n_wvl],  # type: ignore [index]
                sim_spectra=simulation_data["generic_sims"][:, :n_wvl],
                pc_x="PC_0",
                pc_y="PC_1",
                prefix=(
                    f"{cfg['data']['real_data']}_{data_key}_generic_sims{surrogate_suffix}_"
                    f"{cfg['pca']['aggregation_level']}{overlap_suffix}{specular_suffix}"
                ),
                n_examples=10,
            )
            plot_outliers_with_nearest_neighbour(
                transformed_related,
                transformed_sim,
                real_spectra=data[:, :n_wvl],
                sim_spectra=simulation_data["generic_sims"][:, :n_wvl],
                pc_x="PC_0",
                pc_y="PC_1",
                prefix=(
                    f"{data_key}_generic_sims{surrogate_suffix}_"
                    f"{cfg['pca']['aggregation_level']}{overlap_suffix}{specular_suffix}"
                ),
                n_examples=10,
            )


if __name__ == "__main__":
    main()
