"""
Discover and catalog neural scaling model paths and metadata.

This script scans a base directory for model paths, extracts metadata like
training data ratios, and creates a CSV file with model information for use
by evaluation scripts.
"""

import glob
import os
import re

import pandas as pd
from dotenv import load_dotenv

from mcmlnet.utils.logging import setup_logging

load_dotenv()
logger = setup_logging(level="info", logger_name=__name__)


def get_training_data_ratio(path: str) -> float | None:
    """
    Extract training data ratio from model path.

    Args:
        path: Model path string

    Returns:
        Training data ratio as float, or None if not found
    """
    match = re.search(r"tdr_(\d+(\.\d+)?)$", path)
    if match:
        return float(match.group(1))
    logger.warning(f"Could not extract TDR from path: {path}")
    return None


def discover_model_paths(base_path: str, pattern: str) -> list[str]:
    """
    Discover model paths matching the pattern.

    Args:
        base_path: Base directory to search
        pattern: Glob pattern to match model directories

    Returns:
        List of discovered model paths

    Raises:
        ValueError: If no models are found matching the pattern
    """
    search_pattern = os.path.join(base_path, pattern)
    model_paths = glob.glob(search_pattern)

    if not model_paths:
        raise ValueError(f"No models found matching pattern: {search_pattern}")

    logger.info(f"Found {len(model_paths)} model paths")
    return model_paths


def extract_checkpoint_paths(model_path: str) -> str:
    """
    Extract checkpoint paths for a given model path.

    Args:
        model_path: Path to the model directory

    Returns:
        Relative checkpoint path
    """
    checkpoint_dir = os.path.join(model_path, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        logger.warning(f"Checkpoint directory not found: {checkpoint_dir}")
        return ""

    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*ckpt"))
    if not checkpoint_files:
        logger.warning(f"No checkpoint files found in: {checkpoint_dir}")
        return ""

    # Convert to relative paths
    relative_paths = [
        os.path.join("checkpoints", os.path.relpath(cp, checkpoint_dir))
        for cp in checkpoint_files
    ]

    if len(relative_paths) > 1:
        logger.warning(f"Multiple checkpoint files found in: {checkpoint_dir}")
        logger.warning(f"Using first checkpoint file: {relative_paths[0]}")

    return relative_paths[0]


def create_model_metadata_df(model_paths: list[str]) -> pd.DataFrame:
    """
    Create DataFrame with model metadata.

    Args:
        model_paths: List of model paths

    Returns:
        DataFrame with model metadata
    """
    data = []

    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        tdr = get_training_data_ratio(model_path)
        checkpoint_path = extract_checkpoint_paths(model_path)

        data.append(
            {
                "model_path": model_path,
                "model_name": model_name,
                "tdr": tdr,
                "model_checkpoint_path": checkpoint_path,
            }
        )

    return pd.DataFrame(data)


def save_model_metadata(df: pd.DataFrame, output_path: str) -> None:
    """
    Save model metadata to CSV file.

    Args:
        df: DataFrame with model metadata
        output_path: Output file path
    """
    cache_dir = os.getenv("cache_dir")
    if not cache_dir:
        logger.error("CACHE_DIR environment variable not set")
        return

    output_file = os.path.join(cache_dir, output_path)
    df.to_csv(output_file, index=False)
    logger.info(f"Model metadata saved to: {output_file}")


PATTERN = ["*100M_photons_fold*", "*_tdr_*", "*_tdr_*"]
BASE_PATHS = [
    os.path.join(os.environ["data_dir"], "models/scaling_models"),
    os.path.join(os.environ["data_dir"], "models/other_scaling_models"),
    os.path.join(os.environ["data_dir"], "models/related_work"),
]
RESULTS_DIR = os.path.join(os.environ["cache_dir"], "discovered_scaling_models_df.csv")


def main() -> None:
    logger.info("Starting model discovery...")

    model_df = pd.DataFrame()

    for base_path, pattern in zip(BASE_PATHS, PATTERN, strict=False):
        logger.info(f"Searching in: {base_path}")
        logger.info(f"Using pattern: {pattern}")
        model_paths = discover_model_paths(base_path, pattern)
        model_df = pd.concat([model_df, create_model_metadata_df(model_paths)])

    logger.info(f"Created metadata for {len(model_df)} models")
    logger.info(f"Models with TDR: {model_df['tdr'].notna().sum()}")
    logger.info(
        "Models with checkpoints: "
        f"{(model_df['model_checkpoint_path'].str.len() > 0).sum()}"
    )
    save_model_metadata(model_df, RESULTS_DIR)


if __name__ == "__main__":
    main()
