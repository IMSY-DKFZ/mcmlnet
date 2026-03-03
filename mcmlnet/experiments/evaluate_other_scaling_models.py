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


# Dataset loading and preprocessing functions
def load_all_evaluation_datasets() -> dict:
    """Load all specific photon evaluation test datasets upfront."""
    logger.info("Loading all raw datasets...")

    split = "test"
    raw_datasets = {}
    dummy_preprocessor = PreProcessor()

    for n_photons in ["0_1M", "1M", "10M"]:
        data = SimulationDataLoader().load_simulation_data(
            f"raw/base_physio_and_physical_simulations/physiological_ablation_{n_photons}_photons.parquet",
        )
        raw_datasets[f"{n_photons}"] = data[
            dummy_preprocessor.consistent_data_split_ids(data, mode=split)
        ]

    logger.info(f"All datasets loaded! Total datasets: {len(raw_datasets)}")
    return raw_datasets


# Model evaluation functions
def evaluate_model(
    model: pl.LightningModule,
    processed_dataset: torch.Tensor,
    name: str,
    batch_size: int = 1000,
) -> dict[str, float]:
    """Evaluate a model on all datasets."""

    mae = calculate_mae(model, processed_dataset, name, batch_size)
    r2_score = calculate_r2(model, processed_dataset, name, batch_size)
    results = {"mae": mae, "r2_score": r2_score}

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
    evaluation_results: dict[str, float],
) -> None:
    """Add evaluation results for a model."""

    metrics = ["mae", "r2_score"]

    flat_keys = ["tdr", *metrics]
    flat_results = [model_config["tdr"]] + [
        evaluation_results[metric] for metric in metrics
    ]
    results_dict[model_config["name"]] = dict(
        zip(flat_keys, flat_results, strict=False)
    )


if __name__ == "__main__":
    torch.set_printoptions(precision=5)

    # Validate that model metadata file exists and has the correct properties
    validate_model_metadata_file()

    # Initialize results dictionary
    results_mlp_10M: dict[str, dict[str, float]] = {}
    results_mlp_1M: dict[str, dict[str, float]] = {}
    results_mlp_100k: dict[str, dict[str, float]] = {}

    # Load all datasets upfront
    print("=" * 50)
    print("LOADING ALL DATASETS...")
    print("=" * 50)
    raw_datasets = load_all_evaluation_datasets()

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

        if not (
            "MLP_0_1M" in model_config["name"]
            or "MLP_1M" in model_config["name"]
            or "MLP_10M" in model_config["name"]
        ):
            continue

        # Reset logger level to ERROR (instantiate overwrites the logger level)
        configure_all_loggers(GLOBAL_LOGGING_LEVEL)

        # Preprocess datasets for this specific model
        processed_datasets = preprocess_datasets_for_model(
            raw_datasets, preprocessor, cfg
        )

        # Evaluate and store results
        if "MLP_10M" in model_config["name"]:
            evaluation_results = evaluate_model(model, processed_datasets["10M"], "10M")
            add_model_results(results_mlp_10M, model_config, evaluation_results)
        elif "MLP_1M" in model_config["name"]:
            evaluation_results = evaluate_model(model, processed_datasets["1M"], "1M")
            add_model_results(results_mlp_1M, model_config, evaluation_results)
        elif "MLP_0_1M" in model_config["name"]:
            evaluation_results = evaluate_model(
                model, processed_datasets["0_1M"], "0_1M"
            )
            add_model_results(results_mlp_100k, model_config, evaluation_results)
        else:
            raise ValueError(f"Model {model_config['name']} not compatible")

    # Display and save results to CSV file
    output_filename = "10M_mlp_test_results.csv"
    save_results_to_csv(results_mlp_10M, output_filename)
    output_filename = "1M_mlp_test_results.csv"
    save_results_to_csv(results_mlp_1M, output_filename)
    output_filename = "100k_mlp_test_results.csv"
    save_results_to_csv(results_mlp_100k, output_filename)
