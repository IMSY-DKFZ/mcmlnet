"""
Compares the coverage of simulated TIVITA spectral data against multiple real
simulation sets using either cosine similarity or MAE threshold of the nearest
neighbour as inclusion criterion.

This script evaluates how well different simulation datasets can represent
real-world spectral data by computing inclusion based on similarity/distance
thresholds. It supports different aggregation levels (image-level means or
raw spectra) and different distance metrics.
"""

import os
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from dotenv import load_dotenv
from matplotlib.colors import to_rgba
from rich.progress import track
from torch.utils.data import DataLoader

from mcmlnet.experiments.data_loaders.aggregation import (
    aggregate_data_image_level,
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
from mcmlnet.experiments.plotting import (
    construct_filename,
    get_simulation_color,
)
from mcmlnet.experiments.utils import (
    compute_bootstrap_ci,
    fit_knn,
    predict_knn,
)
from mcmlnet.utils.knn_cuml import CuMLKNeighbors
from mcmlnet.utils.load_configs import label_is_organic, load_config, load_label_map
from mcmlnet.utils.metrics import compute_distance_metric

load_dotenv()

DEFAULT_THRESHOLDS = {
    "cos_sim": 0.995,
    "mae": 0.02,
}

if not os.path.exists(os.environ["plot_save_dir"]):
    os.makedirs(os.environ["plot_save_dir"])


def get_metric_config(metric: str, config: dict) -> dict:
    """Get metric-specific configuration parameters.

    Args:
        metric: The metric to use.
        config: The configuration dictionary.

    Returns:
        A dictionary containing the metric-specific configuration parameters.
    """
    if metric == "cos_sim":
        return {
            "thresholds": config["recall"]["cos_thresholds"],
            "default_threshold": DEFAULT_THRESHOLDS["cos_sim"],
            "threshold_key": "cos_threshold",
            "comparison_op": "greater",  # for cosine similarity, higher is better
            "axis_label": "Cos. Sim. Threshold",
            "recall_label": "Recall @ Cos. Sim.",
            "annotation_text": f"Recall @<br>Cos. Sim. {DEFAULT_THRESHOLDS['cos_sim']}",
            "filename_suffix": "cos_sim",
        }
    elif metric == "l1":
        return {
            "thresholds": config["recall"]["mae_thresholds"],
            "default_threshold": DEFAULT_THRESHOLDS["mae"],
            "threshold_key": "mae_threshold",
            "comparison_op": "less",  # for MAE, lower is better
            "axis_label": "MAE Threshold",
            "recall_label": "Recall @ MAE.",
            "annotation_text": f"Recall @<br>MAE {DEFAULT_THRESHOLDS['mae']}",
            "filename_suffix": "mae",
        }
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def compute_recalls(
    knn: CuMLKNeighbors,
    recalls: list,
    sim_data: torch.Tensor,
    sim_name: str,
    real_data: torch.Tensor,
    labels: torch.Tensor,
    subject_ids: list[str] | str,
    config: dict,
) -> None:
    """
    Computes recall values for various thresholds, either at image-level or raw level.

    Args:
        knn: The fitted CuMLKNeighbors model on simulation data.
        recalls: A list to store the computed recall dictionaries.
        sim_data: Tensor containing the simulation data.
        sim_name: Name of the simulation dataset.
        real_data: Tensor containing the real-world spectral data.
        labels: Tensor containing labels for the real_data.
        subject_ids: List of subject IDs corresponding to the real_data
            or a single subject ID.
        config: Dictionary containing configuration parameters, including
                recall settings (thresholds, aggregation_level) and
                data settings (organic_only, real_data).
    """
    # Predict neighbors for real data (fit kNN only once)
    ids_real, _ = predict_knn(knn, real_data, verbose=False)

    # Get unique (organic) labels
    unique_labels = torch.unique(labels).cpu().numpy()
    if config["data"]["organic_only"]:
        organic_mask = label_is_organic(
            unique_labels, dataset=config["data"]["real_data"]
        )
        unique_labels = unique_labels[organic_mask]

    # Collect recall metric computation parameters
    aggregation = config["recall"]["aggregation_level"]
    metric = config["recall"]["metric"]
    metric_config = get_metric_config(metric, config)

    # Compute the distance/similarity for the nearest neighbor
    distance_values = compute_distance_metric(
        real_data, sim_data[ids_real[:, 0]], metric=metric
    )

    for threshold in metric_config["thresholds"]:
        for label in unique_labels:
            # Collect spectral "inliers"
            label_mask = labels == label
            if metric_config["comparison_op"] == "greater":
                is_inlier = distance_values[label_mask] > threshold
            else:
                is_inlier = distance_values[label_mask] < threshold

            # Filter subject_ids based on label_mask
            if isinstance(subject_ids, list):
                filtered_subject_ids = [
                    subject_ids[i] for i in range(len(subject_ids)) if label_mask[i]
                ]
                if len(filtered_subject_ids) != len(is_inlier):
                    raise ValueError(
                        "Length mismatch between subject_ids and is_inlier "
                        f"for label {label}."
                    )
            else:
                filtered_subject_ids = subject_ids  # type: ignore [assignment]

            # Compute recall for image-level mean spectra
            if aggregation == "image":
                recalls.append(
                    {
                        "sim_name": [sim_name] * len(is_inlier),
                        "label": [label.item()] * len(is_inlier),
                        "is_inlier": is_inlier.cpu().tolist(),
                        "subject_ids": filtered_subject_ids,
                        metric_config["threshold_key"]: [threshold] * len(is_inlier),
                    }
                )
            # Compute unnormalized recall for raw (not aggregated) spectra (iteratively)
            elif aggregation == "raw":
                unnormalized_recall = torch.sum(is_inlier)
                recalls.append(
                    {
                        "sim_name": sim_name,
                        "label": label.item(),
                        "unnormalized_recall": unnormalized_recall.item(),
                        "n_spectra": torch.sum(label_mask).item(),
                        "subject_ids": filtered_subject_ids,
                        metric_config["threshold_key"]: threshold,
                    }
                )


def aggregate_recalls(
    recalls: list,
    label_map: dict,
    config: dict[str, dict[str, str | bool | int | list[float]]],
) -> pd.DataFrame:
    """
    Aggregates recall results into a pandas DataFrame and maps label IDs to names.

    If recalls were computed at a "raw" aggregation level (unnormalized), this
    function calculates the final recall by summing unnormalized recalls and total
    spectra counts per group before division.

    Args:
        recalls: A list of dictionaries, where each dictionary contains
                 recall information for a specific label, threshold, and simulation.
        label_map: A dictionary mapping label IDs to human-readable label names.
        config: Configuration dictionary containing metric and data settings.

    Returns:
        A pandas DataFrame with aggregated recall information.
    """
    # Turn list of dicts into a dataframe
    try:
        flattened_recalls = []
        for recall in recalls:
            max_length = max(len(v) for v in recall.values())
            for i in range(max_length):
                flattened_recalls.append({k: v[i] for k, v in recall.items()})
        df = pd.DataFrame(flattened_recalls)
    except TypeError:
        # If recalls is already a list of dicts with the same length, no need to flatten
        df = pd.DataFrame(recalls)

    # Filter labels if organic_only is set
    if config["data"]["organic_only"]:
        unique_labels = df["label"].unique()
        organic_mask = label_is_organic(
            unique_labels,
            dataset=config["data"]["real_data"],  # type: ignore [arg-type]
        )
        df = df[df["label"].isin(unique_labels[organic_mask])]

    # Map label IDs to names
    df["label"] = df["label"].map(label_map)

    # Get metric configuration
    metric_config = get_metric_config(config["recall"]["metric"], config)  # type: ignore [arg-type]
    threshold_key = metric_config["threshold_key"]

    # Compute subject-wise recall from aggregated and raw data
    if "is_inlier" in df.columns:
        df_agg = (
            df.groupby(["label", threshold_key, "sim_name", "subject_ids"])
            .agg({"is_inlier": "mean"})
            .reset_index()
        )
        df_agg["recall_per_id"] = df_agg["is_inlier"]
        df = df_agg
    elif "unnormalized_recall" in df.columns:
        df_agg = (
            df.groupby(["label", threshold_key, "sim_name", "subject_ids"])
            .agg({"unnormalized_recall": "sum", "n_spectra": "sum"})
            .reset_index()
        )
        df_agg["recall_per_id"] = df_agg["unnormalized_recall"] / df_agg["n_spectra"]
        df = df_agg

    return df


def _aggregate_subject_recalls(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Hierarchically aggregate recalls across subjects after computing subject means.

    Args:
        df: DataFrame containing recall data.
        config: Configuration dictionary containing metric and data settings.

    Returns:
        A pandas DataFrame with aggregated recall information.
    """
    metric_config = get_metric_config(config["recall"]["metric"], config)
    threshold_key = metric_config["threshold_key"]

    df = (
        df.groupby(["label", threshold_key, "sim_name"])
        .agg({"recall_per_id": "mean"})
        .reset_index()
    )
    df.rename(columns={"recall_per_id": "recall"}, inplace=True)

    return df


def _get_recall_difference(
    df: pd.DataFrame,
    ref_sim_name: str,
    config: dict,
    threshold: float,
) -> None:
    """
    Get the recall difference for a specific reference simulation name and threshold.

    Args:
        df: DataFrame containing recall data.
        ref_sim_name: Name of the reference simulation.
        config: Configuration dictionary containing metric and data settings.
        threshold: Threshold value to filter.

    Returns:
        None
    """
    metric = config["recall"]["metric"]
    metric_config = get_metric_config(metric, config)
    threshold_key = metric_config["threshold_key"]

    # Filter the DataFrame for the specified simulation name and threshold
    df = df.copy()[df[threshold_key] == threshold]
    reference_df = df[df["sim_name"] == ref_sim_name].copy()

    # Merge the DataFrame with the reference DataFrame for subtraction
    merged_df = df.merge(
        reference_df,
        on=["label"],
        suffixes=("", f"_{ref_sim_name}"),
    )
    merged_df["recall_difference"] = (
        merged_df["recall"] - merged_df[f"recall_{ref_sim_name}"]
    )

    # Filter out the reference simulation rows
    result_df = merged_df[merged_df["sim_name"] != ref_sim_name]

    # Print grouped by simulation name
    for sim_name, group in result_df.groupby("sim_name"):
        print(f"Recall difference for {sim_name} at {metric} {threshold}:")
        print(
            group[
                [
                    "label",
                    "recall",
                    f"recall_{ref_sim_name}",
                    "recall_difference",
                ]
            ]
        )
        print(
            f"Mean recall difference: {group['recall_difference'].mean():.4f} "
            f"(std: {group['recall_difference'].std():.4f})"
        )
        ci_lower, ci_upper = compute_bootstrap_ci(
            group["recall_difference"].values, 0.95
        )
        print(f"95% confidence interval: [{ci_lower:.4f}, {ci_upper:.4f}]")


def plot_recall_spider_plot(
    df: pd.DataFrame,
    config: dict,
) -> None:
    """
    Plot a spider plot (radar chart) for recall metrics
    grouped by labels and simulations.

    Args:
        df: DataFrame containing recall data.
        config: Configuration dictionary containing metric and data settings.

    Returns:
        None
    """
    metric_config = get_metric_config(config["recall"]["metric"], config)

    df = _aggregate_subject_recalls(df, config)

    # Show difference
    _get_recall_difference(
        df, "generic_sims", config, metric_config["default_threshold"]
    )

    # Get unique labels and simulations
    labels = df["label"].unique()
    labels = [label.replace("_", " ") for label in labels]
    simulations = df["sim_name"].unique()

    # Initialize the figure
    fig = go.Figure()

    simulation_order = [
        "generic_sims",
        "lan_sims",
        "jacques_sims",
        "manoj_sims",
        "tsui_sims",
        "generic_sims_8400k",
        "generic_sims_420k",
        "generic_sims_21k",
    ]

    # Add a trace for each simulation
    for sim_name in simulation_order:
        if sim_name not in simulations:
            continue
        df_sim = df[df["sim_name"] == sim_name].copy()
        for _i, threshold in enumerate(metric_config["thresholds"]):
            # filter by threshold
            df_thresh = df_sim[
                df_sim[metric_config["threshold_key"]] == threshold
            ].copy()
            values = df_thresh["recall"].values.tolist()

            # Add a radar trace
            fig.add_trace(
                go.Scatterpolar(
                    r=[*values, values[0]],
                    theta=[*labels, labels[0]],
                    fill="toself",
                    name=f"{DataConfig.SIMULATION_NAMES[sim_name]}",
                    marker={"color": get_simulation_color(sim_name), "size": 3},
                    line={"color": get_simulation_color(sim_name), "width": 1},
                    visible=(
                        False
                        if threshold != metric_config["default_threshold"]
                        else True
                    ),
                    showlegend=(
                        False
                        if threshold != metric_config["default_threshold"]
                        else True
                    ),
                )
            )

    # update layout
    fig.update_polars(hole=0.2)  # hollow center (20%)
    fig.update_layout(
        polar={
            "angularaxis": {"rotation": 90, "direction": "clockwise", "showgrid": True},
            "radialaxis": {"visible": True, "range": [0, 1]},
        },
        showlegend=True,
        legend={
            "title": "Simulation",
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
        },
        template="presentation",
        font={
            "family": "Times New Roman",
            "size": 10,
            "color": "black",
        },
    )

    # Add radial axis label
    fig.add_annotation(
        text=metric_config["annotation_text"],
        x=0.5,
        y=0.5,
        showarrow=False,
    )

    # Construct filename and save plot
    test_suffix = (
        f"_{config['data']['test_data']}" if config["data"]["test_data"] else ""
    )
    output_path = construct_filename(
        config, f"recall_spider_{metric_config['filename_suffix']}{test_suffix}"
    )
    fig.write_image(output_path, width=470, height=430)
    print(f"Saved spider plot to {output_path}")


def plot_recall_convergence_lineplot(
    df: pd.DataFrame,
    config: dict,
) -> None:
    """
    Plot recall metrics as a line plot with threshold on the x-axis
    to assess convergence properties.

    Args:
        df: DataFrame containing recall data.
        config: Configuration dictionary containing metric and data settings.

    Returns:
        None
    """
    metric = config["recall"]["metric"]
    metric_config = get_metric_config(metric, config)

    df = _aggregate_subject_recalls(df, config)
    fig = go.Figure()

    # Add a trace for each simulation
    threshold_values = np.sort(df[metric_config["threshold_key"]].unique())
    for sim_name in df["sim_name"].unique():
        df_sim = df[df["sim_name"] == sim_name].copy()

        # Group by thresholds and compute quantiles
        grouped = df_sim.groupby(metric_config["threshold_key"])
        q1 = grouped["recall"].quantile(0.25)
        q3 = grouped["recall"].quantile(0.75)
        mean = grouped["recall"].mean()
        median = grouped["recall"].median()

        fig.add_trace(
            go.Scatter(
                x=threshold_values,
                y=q1.tolist(),
                mode="lines",
                fill=None,
                line={"color": get_simulation_color(sim_name), "width": 0.5},
                legendgroup=f"{DataConfig.SIMULATION_NAMES[sim_name]}",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=threshold_values,
                y=q3.tolist(),
                mode="lines",
                fill="tonexty",
                fillcolor="rgba"
                + str(to_rgba(get_simulation_color(sim_name), alpha=0.3)),
                line={"color": get_simulation_color(sim_name), "width": 0.5},
                name=f"{DataConfig.SIMULATION_NAMES[sim_name]} (IQR Across Organs)",
                legendgroup=f"{DataConfig.SIMULATION_NAMES[sim_name]}",
                visible=True,
                showlegend=True,
            )
        )
        # Add median trace
        fig.add_trace(
            go.Scatter(
                x=threshold_values,
                y=median,
                mode="lines+markers",
                line={
                    "color": get_simulation_color(sim_name),
                    "dash": "dash",
                },
                marker={"size": 8},
                name=f"{DataConfig.SIMULATION_NAMES[sim_name]} (Median)",
                legendgroup=f"{DataConfig.SIMULATION_NAMES[sim_name]}",
            )
        )

        # Add mean trace
        fig.add_trace(
            go.Scatter(
                x=threshold_values,
                y=mean,
                mode="lines+markers",
                line={
                    "color": get_simulation_color(sim_name),
                    "dash": "dot",
                },
                marker={"size": 8},
                name=f"{DataConfig.SIMULATION_NAMES[sim_name]} (Mean)",
                legendgroup=f"{DataConfig.SIMULATION_NAMES[sim_name]}",
            )
        )

    # Update layout
    x_axis_config = {
        "title": metric_config["axis_label"],
        "showgrid": True,
        "gridcolor": "lightgray",
        "gridwidth": 0.5,
    }

    # Add log scale for MAE
    if metric == "l1":
        x_axis_config["type"] = "log"

    fig.update_layout(
        xaxis=x_axis_config,
        yaxis={
            "title": metric_config["recall_label"],
            "showgrid": True,
            "gridcolor": "lightgray",
            "gridwidth": 0.5,
        },
        showlegend=True,
        legend={"title": "Simulation"},
        title=(
            "Recall/ Coverage Convergence for Increasing "
            f"{metric_config['axis_label']}s"
        ),
        template="presentation",
        font={
            "family": "Times New Roman",
            "size": 8,
            "color": "black",
        },
    )

    # Construct filename and save plot
    test_suffix = (
        f"_{config['data']['test_data']}" if config["data"]["test_data"] else ""
    )
    output_path = construct_filename(
        config,
        f"recall_convergence_lineplot_{metric_config['filename_suffix']}{test_suffix}",
    )
    fig.write_image(output_path, width=4.5 * 200, height=2.5 * 200)
    print(f"Saved line plot to {output_path}")


def main() -> None:
    """
    Main function to execute the recall computation and analysis pipeline.

    Steps:
    1. Loads configuration and label maps.
    2. Checks for cached recall results; otherwise, proceeds with computation.
    3. Loads real-world spectral data and simulation datasets.
    4. For each simulation dataset:
        a. Fits a kNN model.
        b. Computes distances within the simulation data.
        c. Computes recalls against real data, either at image or raw level.
    5. Caches the computed recalls.
    6. Aggregates recall results.
    7. Plots recall spider plots for different thresholds.
    8. Plot recalls as line plots for each simulation dataset and label to check
       convergence with threshold.
    """
    # Load config and label map
    cfg = load_config()
    label_map = load_label_map(dataset=cfg["data"]["real_data"])

    # Get metric configuration
    metric = cfg["recall"]["metric"]
    metric_config = get_metric_config(metric, cfg)

    # Cache results
    specular_suffix = "_specular" if cfg["data"]["specular"] else ""
    overlap_suffix = "_minimal_overlap" if cfg["recall"]["minimal_overlap"] else ""
    surrogate_suffix = "_surrogate" if cfg["recall"]["use_surrogate_model"] else ""
    ablation_suffix = (
        "_ablation" if cfg["recall"]["do_surrogate_model_ablation"] else ""
    )
    test_suffix = f"_{cfg['data']['test_data']}" if cfg["data"]["test_data"] else ""
    cache_path = os.path.join(
        os.environ["cache_dir"],
        f"{cfg['data']['real_data']}_data_reflectance{surrogate_suffix}{ablation_suffix}_"
        f"{cfg['recall']['metric']}_recall_{metric_config['filename_suffix']}{specular_suffix}"
        f"_{cfg['recall']['aggregation_level']}{overlap_suffix}{test_suffix}.pkl",
    )

    try:
        recalls = pickle.load(open(cache_path, "rb"))
    except FileNotFoundError:
        # Collect real data
        dataloader: DataLoader
        if cfg["data"]["real_data"] == "human":
            dataloader = CombinedHumanDataLoader()
        else:
            dataloader = PigDataLoader()

        if cfg["data"]["test_data"] == "test":
            dataloader = dataloader.load_test_data()
        elif cfg["data"]["test_data"] == "final":
            dataloader = dataloader.load_final_data()
        else:
            dataloader, _ = dataloader.load_training_data()

        if cfg["recall"]["aggregation_level"] == "image":
            aggregated_data = aggregate_data_image_level(
                dataloader,
                cfg["data"]["real_data"],
                cfg["data"]["organic_only"],
            )
        elif cfg["recall"]["aggregation_level"] == "raw":
            aggregated_data = None
        else:
            raise ValueError(
                f"Unknown aggregation level: {cfg['recall']['aggregation_level']}"
            ) from None

        # Load adapted simulation datasets
        if cfg["recall"]["do_surrogate_model_ablation"] is True:
            print("Running recall inference with ablation surrogate model data ...")
            simulation_data = SimulationDataLoaderManager(
                cfg["data"]["specular"]
            ).load_surrogate_model_ablation_data()
        else:
            if cfg["recall"]["use_surrogate_model"] is True:
                simulation_data = SimulationDataLoaderManager(
                    cfg["data"]["specular"]
                ).load_surrogate_model_data()
            else:
                simulation_data = SimulationDataLoaderManager(
                    cfg["data"]["specular"]
                ).load_simulation_data()

        recalls = []

        for sim_name, sim_data in track(
            simulation_data.items(), description="Computing recalls"
        ):
            # Intermediate result caching
            cache_path_intermed = cache_path.replace(".pkl", f"_{sim_name}.pkl")
            print(f"Caching collected recall dataset ot {cache_path_intermed}")
            try:
                _recall = pickle.load(open(cache_path_intermed, "rb"))
            except FileNotFoundError:
                print(f"Computing recalls for {sim_name}...")
                _recall = []
                # Choose to only use the global intersection of all simulation data
                if cfg["recall"]["minimal_overlap"] is True:
                    n_wvl = np.min(
                        [_data.shape[1] for _data in simulation_data.values()]
                    )
                else:
                    n_wvl = sim_data.shape[1]
                # Fit kNN
                knn = fit_knn(sim_data[:, :n_wvl], k=1, distance_type=metric)

                if aggregated_data is not None:
                    # Collect the real data
                    labels = torch.tensor(aggregated_data["label_ids"])
                    subject_ids = aggregated_data["subject_ids"]
                    mean_spectra = torch.from_numpy(aggregated_data["mean_spectra"])
                    compute_recalls(
                        knn,
                        _recall,
                        sim_data[:, :n_wvl],
                        sim_name,
                        mean_spectra[:, :n_wvl],
                        labels,
                        subject_ids,
                        cfg,
                    )
                else:
                    # Iterate over the images
                    for _i, batch in track(
                        enumerate(dataloader),
                        total=len(dataloader),
                        description="Computing recall per image",
                        transient=True,
                    ):
                        # Load and clamp spectra to physically meaningful values
                        img = torch.clamp(batch["features"], 0, 1)
                        img = img.view(-1, img.shape[-1])
                        labels = batch["labels"].view(-1)
                        # Get subject ID string
                        if cfg["data"]["real_data"] in ["pig_semantic", "pig_masks"]:
                            subject_id = batch["image_name"][0][:4]
                        else:
                            subject_id = batch["image_name"][0][:12]
                        compute_recalls(
                            knn,
                            _recall,
                            sim_data[:, :n_wvl],
                            sim_name,
                            img[:, :n_wvl],
                            labels,
                            subject_id,
                            cfg,
                        )
                # Cache intermediate results
                pickle.dump(_recall, open(cache_path_intermed, "wb"))
            # Concatenate intermediate results
            recalls.extend(_recall)
        pickle.dump(recalls, open(cache_path, "wb"))

    # Process and plot results
    df = aggregate_recalls(recalls, label_map, cfg)

    # Plot results
    plot_recall_spider_plot(
        df,
        cfg,
    )
    plot_recall_convergence_lineplot(
        df,
        cfg,
    )


if __name__ == "__main__":
    main()
