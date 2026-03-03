import logging

import lightning as pl
import torch
from rich.progress import track

from mcmlnet.experiments.utils import (
    load_model_configs,
    save_results_to_csv,
    validate_model_metadata_file,
)
from mcmlnet.training.data_loading.sota_data_classes import SOTAPreprocessor
from mcmlnet.training.models.base_model import BaseModel
from mcmlnet.utils.convenience import (
    load_trained_model,
    predict_in_batches,
)
from mcmlnet.utils.logging import configure_all_loggers, setup_logging

GLOBAL_LOGGING_LEVEL = logging.ERROR
configure_all_loggers(GLOBAL_LOGGING_LEVEL)
logger = setup_logging(level="warning", logger_name=__name__)


# Dataset loading and preprocessing functions
def load_related_work_preprocessed_test_dataset(dataset_name: str) -> torch.Tensor:
    """Load related work preprocessed test datasets."""
    if dataset_name not in ["manoj", "lan", "lan_lhs", "tsui"]:
        raise ValueError(
            "dataset_name must be either 'manoj', 'lan', 'lan_lhs', or 'tsui'"
        )

    if dataset_name == "manoj":
        preprocessor = SOTAPreprocessor(dataset_name, n_wavelengths=351)
        data = preprocessor.fit()
    elif dataset_name == "lan_lhs" or dataset_name == "lan":
        preprocessor = SOTAPreprocessor("lan_lhs", n_wavelengths=1)
        data = preprocessor.fit()
    elif dataset_name == "tsui":
        preprocessor = SOTAPreprocessor(dataset_name, n_wavelengths=1)
        data = preprocessor.fit()

    test_ids = preprocessor.consistent_data_split_ids(data, mode="test")
    test_data = preprocessor(data[test_ids])
    test_data[..., -2] = 0

    return test_data


# Model evaluation functions
def evaluate_model(
    model: pl.LightningModule,
    processed_dataset: dict,
    batch_size: int = 1000,
) -> dict[str, float]:
    """Evaluate a model on a processed dataset."""
    mae = calculate_mae(model, processed_dataset, "related_work", batch_size)

    return {"mae": mae}


def calculate_mae(
    model: pl.LightningModule, data: torch.Tensor, name: str, batch_size: int = 1000
) -> float:
    """Calculate MAE for model predictions."""
    y_pred = predict_in_batches(model, data[..., :-1], batch_size=batch_size).squeeze()
    y_true = data[..., -1].squeeze()

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


# Results management functions
def add_model_results(
    results_dict: dict[str, dict[str, float]],
    model_config: dict,
    evaluation_results: dict[str, float],
) -> None:
    """Add evaluation results for a model."""
    results_dict[model_config["name"]] = {
        "tdr": model_config["tdr"],
        "mae": evaluation_results["mae"],
    }


if __name__ == "__main__":
    torch.set_printoptions(precision=5)

    # Validate that model metadata file exists and has the correct properties
    validate_model_metadata_file()

    # Initialize results dictionary
    results_manoj: dict[str, dict[str, float]] = {}
    results_lan: dict[str, dict[str, float]] = {}
    results_tsui: dict[str, dict[str, float]] = {}

    # Load all datasets upfront
    print("=" * 50)
    print("LOADING ALL DATASETS...")
    print("=" * 50)

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
        # Include related work models
        if not (
            "manoj" in model_config["name"]
            or "lan" in model_config["name"]
            or "tsui" in model_config["name"]
        ):
            continue

        # Load model and preprocessor
        model, preprocessor, cfg = load_trained_model(
            model_config["path"], model_config["checkpoint_path"], BaseModel
        )

        # Reset logger level to ERROR (instantiate overwrites the logger level)
        configure_all_loggers(GLOBAL_LOGGING_LEVEL)

        # Preprocess datasets for this specific model
        dataset_name = model_config["name"].split("_")[0]
        processed_dataset = load_related_work_preprocessed_test_dataset(dataset_name)

        # Evaluate model on all datasets
        evaluation_results = evaluate_model(model, processed_dataset)

        # Store results
        if "manoj" in model_config["name"]:
            add_model_results(results_manoj, model_config, evaluation_results)
        elif "lan" in model_config["name"]:
            add_model_results(results_lan, model_config, evaluation_results)
        elif "tsui" in model_config["name"]:
            add_model_results(results_tsui, model_config, evaluation_results)
        else:
            raise ValueError(f"Model {model_config['name']} not found")

    # Display and save results to CSV file
    output_filename = "manoj_results.csv"
    save_results_to_csv(results_manoj, output_filename)
    output_filename = "lan_results.csv"
    save_results_to_csv(results_lan, output_filename)
    output_filename = "tsui_results.csv"
    save_results_to_csv(results_tsui, output_filename)
