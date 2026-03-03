import argparse
import logging

import lightning as pl
import torch
from rich.progress import track

from mcmlnet.experiments.utils import (
    load_model_configs,
    preprocess_datasets_for_model,
    save_results_to_csv,
    validate_model_metadata_file,
)
from mcmlnet.training.data_loading.preprocessing import (
    PreProcessor,
)
from mcmlnet.training.models.base_model import BaseModel
from mcmlnet.training.models.kan_model import KANModel
from mcmlnet.utils.convenience import (
    load_trained_model,
    predict_in_batches,
)
from mcmlnet.utils.loading import (
    SimulationDataLoader,
)
from mcmlnet.utils.logging import configure_all_loggers, setup_logging
from mcmlnet.utils.metrics import (
    custom_r2_score,
)

GLOBAL_LOGGING_LEVEL = logging.ERROR
configure_all_loggers(GLOBAL_LOGGING_LEVEL)
logger = setup_logging(level="warning", logger_name=__name__)

N_WVL_PHYSICAL = 351


# Dataset loading and preprocessing functions
def get_10x_photon_data(split: str, param_type: str) -> torch.Tensor:
    """Get 10x photon data."""
    if param_type not in ["physiological", "physical"]:
        raise ValueError("param_type must be either 'physiological' or 'physical'")
    if split not in ["val", "test"]:
        raise ValueError("split must be either 'val' or 'test'")

    if param_type == "physiological":
        file_path = (
            "raw/base_physio_and_physical_simulations/"
            f"physiological_training_100M_1M_samples_{split}_10x_photons.parquet"
        )
    elif param_type == "physical":
        file_path = (
            "raw/base_physio_and_physical_simulations/"
            f"physical_generalization_100M_{split}_10x_photons.parquet"
        )

    if param_type == "physiological":
        data = SimulationDataLoader().load_data(file_path)
    elif param_type == "physical":
        data = SimulationDataLoader().load_physical_simulation_data(
            file_path, n_wavelengths=N_WVL_PHYSICAL
        )

    return data


def load_evaluation_dataset(
    reference_set: str, sampling_style: str, data_split: str
) -> torch.Tensor:
    """Load raw dataset without preprocessing/normalization."""
    if reference_set not in ["normal", "10x"]:
        raise ValueError("reference_set must be either 'normal' or '10x'")
    if sampling_style not in ["physiological", "physical"]:
        raise ValueError("sampling_style must be either 'physiological' or 'physical'")
    if data_split not in ["val", "test"]:
        raise ValueError("data_split must be either 'val' or 'test'")

    dummy_preprocessor = PreProcessor()

    if reference_set == "normal":
        if sampling_style == "physiological":
            data = SimulationDataLoader().load_simulation_data(
                "raw/base_physio_and_physical_simulations/physiological_training_100M_photons_1M_samples.parquet",
            )
            if data_split == "val":
                data = data[
                    dummy_preprocessor.consistent_data_split_ids(data, mode="val")
                ]
            elif data_split == "test":
                data = data[
                    dummy_preprocessor.consistent_data_split_ids(data, mode="test")
                ]
        elif sampling_style == "physical":
            data = SimulationDataLoader().load_physical_simulation_data(
                "raw/base_physio_and_physical_simulations/physical_generalization_100M_photons.parquet",
                n_wavelengths=N_WVL_PHYSICAL,
            )
            if data_split == "val":
                data = data[
                    dummy_preprocessor.consistent_data_split_ids(data, mode="val")
                ]
            elif data_split == "test":
                data = data[
                    dummy_preprocessor.consistent_data_split_ids(data, mode="test")
                ]
    elif reference_set == "10x":
        data = get_10x_photon_data(data_split, sampling_style)

    logger.info(
        f"Loaded raw {sampling_style} {reference_set} {data_split} data "
        f"with shape: {data.shape}"
    )
    return data


def load_all_evaluation_datasets(reference_set: str) -> dict:
    """Load all evaluation datasets upfront."""
    logger.info("Loading all raw datasets...")

    raw_datasets = {}
    sampling_styles = ["physical", "physiological"]
    if reference_set == "10x":
        splits = ["test"]
    else:
        splits = ["val", "test"]

    for sampling_style in sampling_styles:
        for split in splits:
            key = f"{sampling_style}_{split}"
            logger.info(f"Loading {key} dataset...")
            raw_datasets[key] = load_evaluation_dataset(
                reference_set, sampling_style, split
            )

    logger.info(f"All datasets loaded! Total datasets: {len(raw_datasets)}")
    return raw_datasets


# Model evaluation functions
def evaluate_model(
    model: pl.LightningModule,
    processed_datasets: dict,
    reference_set: str,
    batch_size: int = 1000,
) -> dict:
    """Evaluate a model on all datasets."""
    results = {}

    keys = processed_datasets.keys()
    sampling_styles = list({key.split("_")[0] for key in keys})
    splits = list({key.split("_")[1] for key in keys})

    for sampling_style in sampling_styles:
        for split in splits:
            key = f"{sampling_style}_{split}"
            data = processed_datasets[key]

            mae = calculate_mae(
                model, data, f"{sampling_style} {reference_set} {split}", batch_size
            )
            r2_score = None

            if sampling_style == "physiological":
                r2_score = calculate_r2(
                    model, data, f"physiological {split}", batch_size
                )

            results[key] = {"mae": mae, "r2_score": r2_score}

    return results


def calculate_mae(
    model: pl.LightningModule, data: torch.Tensor, name: str, batch_size: int = 1000
) -> float:
    """Calculate MAE for model predictions."""
    y_pred = predict_in_batches(model, data[..., :-1], batch_size=batch_size).squeeze()
    y_true = data[..., -1].reshape(-1, y_pred.shape[-1])

    logger.info(
        f"Min/Max reflectance values: {y_true.min().item():.5f}, "
        f"{y_true.max().item():.5f}"
    )
    logger.info(
        f"Min/Max predicted reflectance values: {y_pred.min().item():.5f}, "
        f"{y_pred.max().item():.5f}"
    )

    mae = torch.nn.functional.l1_loss(y_pred, y_true).item()
    logger.info(f"{name} Data MAE: {mae:.5f}")
    return float(mae)


def calculate_r2(
    model: pl.LightningModule, data: torch.Tensor, name: str, batch_size: int = 1000
) -> float:
    """Calculate R² score for model predictions."""
    y_pred = predict_in_batches(model, data[..., :-1], batch_size=batch_size).squeeze()
    y_true = data[..., -1].reshape(-1, y_pred.shape[-1])

    r2_score = custom_r2_score(y_true, y_pred).mean().item()
    print(f"{name} Data R² score: {r2_score:.5f}")
    return float(r2_score)


# Results management functions
def add_model_results(
    results_dict: dict[str, dict[str, float]],
    model_config: dict,
    evaluation_results: dict[str, dict[str, float]],
) -> None:
    """Add evaluation results for a model."""
    keys = evaluation_results.keys()
    sampling_styles = list({key.split("_")[0] for key in keys})
    splits = list({key.split("_")[1] for key in keys})

    metrics = ["mae", "r2_score"]

    flat_keys = ["tdr"] + [
        f"{metric}_{split}_{sampling_style}"
        for sampling_style in sampling_styles
        for split in splits
        for metric in metrics
    ]

    flat_results = [model_config["tdr"]] + [
        evaluation_results[f"{sampling_style}_{split}"][metric]
        for sampling_style in sampling_styles
        for split in splits
        for metric in metrics
    ]

    results_dict[model_config["name"]] = dict(
        zip(flat_keys, flat_results, strict=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Load surrogate models and evaluate their performance "
            "on validation and test datasets."
        )
    )
    parser.add_argument(
        "--reference_set",
        type=str,
        choices=["normal", "10x"],
        default="10x",
        help="Dataset to evaluate the model on (default: '10x')",
    )
    args = parser.parse_args()

    torch.set_printoptions(precision=5)

    # Validate that model metadata file exists and has the correct properties
    validate_model_metadata_file()

    # Initialize results dictionary
    results: dict[str, dict[str, float]] = {}
    results_kan: dict[str, dict[str, float]] = {}
    results_mlp_10M: dict[str, dict[str, float]] = {}
    results_mlp_1M: dict[str, dict[str, float]] = {}
    results_mlp_100k: dict[str, dict[str, float]] = {}

    # Load all datasets upfront
    print("=" * 50)
    print("LOADING ALL DATASETS...")
    print("=" * 50)
    raw_datasets = load_all_evaluation_datasets(args.reference_set)

    # Load model configurations
    model_configs = load_model_configs()

    # Main evaluation loop
    print("\n" + "=" * 50)
    print("EVALUATING MODELS...")
    print("=" * 50)

    for model_config in track(model_configs, description="Evaluating models"):
        logger.info(
            f"Loading model {model_config['name']} with TDR "
            f"{model_config['tdr']} from {model_config['path']}"
        )
        logger.info(
            f"Base path: {model_config['path']}, Checkpoint: "
            f"{model_config['checkpoint_path']}"
        )

        # Skip related work models
        if (
            "manoj" in model_config["name"]
            or "lan" in model_config["name"]
            or "tsui" in model_config["name"]
        ):
            continue

        # Load model and preprocessor
        try:
            model, preprocessor, cfg = load_trained_model(
                model_config["path"], model_config["checkpoint_path"], BaseModel
            )
        except TypeError:
            model, preprocessor, cfg = load_trained_model(
                model_config["path"], model_config["checkpoint_path"], KANModel
            )

        # Reset logger level to ERROR (instantiate overwrites the logger level)
        configure_all_loggers(GLOBAL_LOGGING_LEVEL)

        # Preprocess datasets for this specific model
        processed_datasets = preprocess_datasets_for_model(
            raw_datasets, preprocessor, cfg
        )

        # Evaluate model on all datasets
        evaluation_results = evaluate_model(
            model, processed_datasets, args.reference_set
        )

        # Store results
        if "KAN_100M" in model_config["name"]:
            add_model_results(results_kan, model_config, evaluation_results)
        elif "MLP_10M" in model_config["name"]:
            add_model_results(results_mlp_10M, model_config, evaluation_results)
        elif "MLP_1M" in model_config["name"]:
            add_model_results(results_mlp_1M, model_config, evaluation_results)
        elif "MLP_0_1M" in model_config["name"]:
            add_model_results(results_mlp_100k, model_config, evaluation_results)
        else:
            add_model_results(results, model_config, evaluation_results)

    # Display and save results to CSV file
    output_filename = f"100M_kan_{args.reference_set}_results.csv"
    save_results_to_csv(results_kan, output_filename)
    output_filename = f"10M_mlp_{args.reference_set}_results.csv"
    save_results_to_csv(results_mlp_10M, output_filename)
    output_filename = f"1M_mlp_{args.reference_set}_results.csv"
    save_results_to_csv(results_mlp_1M, output_filename)
    output_filename = f"100k_mlp_{args.reference_set}_results.csv"
    save_results_to_csv(results_mlp_100k, output_filename)
    output_filename = f"100M_surrogate_model_kfold_{args.reference_set}_results.csv"
    save_results_to_csv(results, output_filename)
