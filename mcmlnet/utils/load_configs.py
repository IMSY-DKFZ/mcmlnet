"""Configuration and utility functions for loading and processing data."""

import os
from typing import Any

import numpy as np
import pandas as pd
import yaml  # type: ignore[import-untyped]
from dotenv import load_dotenv

from mcmlnet.utils.logging import setup_logging

load_dotenv()
logger = setup_logging(level="info", logger_name=__name__)


def get_repo_base_dir() -> str:
    """Get the repository's base directory where experiment_config.yaml is located.

    Returns:
        str: Absolute path to the repository's base directory
    """
    # Start from the current file's directory
    # and walk up until we find experiment_config.yaml
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Walk up the directory tree until we find experiment_config.yaml
    while current_dir != os.path.dirname(current_dir):  # Stop at root
        config_path = os.path.join(current_dir, "experiment_config.yaml")
        if os.path.exists(config_path):
            logger.info(f"Found experiment_config.yaml at: {current_dir}")
            return current_dir
        current_dir = os.path.dirname(current_dir)

    # If we reach here, the config file wasn't found
    raise FileNotFoundError(
        "experiment_config.yaml not found. "
        "Please ensure it exists in the repository root."
    )


def load_config() -> Any:
    """Load the experiment configuration file (dictionary)."""
    repo_base_dir = get_repo_base_dir()
    config_path = os.path.join(repo_base_dir, "experiment_config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(os.path.expandvars(f.read()))
        logger.info(f"Loaded config file: {config_path}")
        logger.info(f"Config:\n{yaml.dump(cfg, indent=2)}")
    return cfg


def load_label_map(dataset: str) -> dict[int, str]:
    """Load the label map for the Tivita dataset.

    Args:
        dataset: The dataset name ("human").

    Returns:
        A dictionary mapping label IDs to human-readable label names.

    Raises:
        ValueError: If the dataset is not recognized.
    """
    if dataset != "human":
        raise ValueError(f"Unknown dataset: {dataset}. Supported: 'human'.")
    # Hard-coded label maps for human dataset
    label_map = {
        "arteries": 0,
        "background": 1,
        "bladder": 2,
        "blood": 3,
        "blue_cloth": 4,
        "bone": 5,
        "cauterization": 6,
        "colon": 7,
        "diaphragm": 8,
        "esophagus": 9,
        "fat": 10,
        "fat_subcutaneous": 11,
        "fat_visceral": 12,
        "gallbladder": 13,
        "heart": 14,
        "instrument": 15,
        "kidney": 16,
        "lig_teres_hep": 17,
        "liver": 18,
        "lung": 19,
        "major_vein": 20,
        "meso": 21,
        "muscle": 22,
        "not_suitable_for_semantic": 23,
        "omentum": 24,
        "pancreas": 25,
        "peritoneum": 26,
        "reflection": 27,
        "skin": 28,
        "small_bowel": 29,
        "spleen": 30,
        "stomach": 31,
        "tag_blood": 32,
        "tag_cauterization": 33,
        "tag_malperfused": 34,
        "tag_tumor": 35,
        "unclear_organic": 36,
        "overlap": 254,
        "unlabeled": 255,
    }
    inv_label_map = {v: k for k, v in label_map.items()}

    # transform labels to publication-friendly format
    def transform_label(label: str) -> str:
        label = label.replace("_", " ")
        if label == "arteries":
            label = "artery"
        elif label == "cauterization":
            label = "cauterized tissue"
        elif label == "fat subcutaneous":
            label = "fat (subcutaneous)"
        elif label == "fat visceral":
            label = "fat (visceral)"
        elif label == "lig teres hep":
            label = "hepatic ligament"
        return label

    transformed_label_map = {k: transform_label(v) for k, v in inv_label_map.items()}

    logger.info(
        f"Loaded {dataset} label map:\n{yaml.dump(transformed_label_map, indent=2)}."
    )

    return transformed_label_map


def label_is_organic(
    label_data: list | np.ndarray | pd.Series, dataset: str
) -> np.ndarray:
    """Filter for organic labels only.

    Args:
        label_data: List, NumPy array, or Pandas Series of labels.
        dataset: The dataset name ("pig_masks", "pig_semantic", or "human").

    Returns:
        A NumPy array of booleans indicating whether each label is organic.
    """
    organic_label_ids_map = {
        "human": [
            0,
            2,
            3,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            16,
            17,
            18,
            19,
            20,
            22,
            24,
            25,
            26,
            28,
            29,
            30,
            31,
        ],
    }

    if dataset not in organic_label_ids_map:
        raise ValueError(f"Unknown dataset: {dataset}.")

    organic_label_ids = organic_label_ids_map[dataset]

    # Convert label_data to a NumPy array if it's not already
    label_data = np.asarray(label_data)

    # Use numpy's isin function to check for organic labels
    return np.isin(label_data, organic_label_ids)


def get_split_name(dataset: str) -> str:
    """Get the appropriate split name based on the dataset."""
    split_map = {
        "pig_masks": "fold_0",
        "pig_semantic": "fold_P044,P050,P059",
        "human": "fold_0",
    }
    if dataset not in split_map:
        raise ValueError(f"Unknown dataset: {dataset}")
    return split_map[dataset]
