"""Utility functions for data loading operations."""

from typing import Any

import numpy as np

from mcmlnet.utils.tensor import (
    TensorType,
    array_conversion_decorator,
)


def subsample_data(
    data: TensorType, subsample_size: int, random_seed: int = 42
) -> TensorType:
    """Subsample data to a specified size.

    Args:
        data: Input data array.
        subsample_size: Target size for subsampling.
        random_seed: Random seed for reproducibility.

    Returns:
        Subsampled data array.

    Raises:
        ValueError: If subsample_size is larger than data size.
    """
    if len(data) <= subsample_size:
        return data

    if subsample_size <= 0:
        raise ValueError(f"Subsample size must be positive, got {subsample_size}")

    np.random.seed(random_seed)
    indices = np.random.choice(len(data), size=subsample_size, replace=False)
    return data[indices]


def get_subject_id(image_name: str, dataset: str) -> str:
    """Extract subject ID from image name based on dataset type.

    Args:
        image_name: Name of the image file.
        dataset: Type of dataset ('pig_semantic', 'pig_masks', or 'human').

    Returns:
        Extracted subject ID.

    Raises:
        ValueError: If dataset type is not supported.
    """
    if dataset in ["pig_semantic", "pig_masks"]:
        return image_name[:4]
    elif dataset == "human":
        return image_name[:12]
    else:
        raise ValueError(f"Unsupported dataset type: {dataset}")


def get_image_id(image_name: str, dataset: str) -> str:
    """Extract image ID from image name based on dataset type.

    Args:
        image_name: Name of the image file.
        dataset: Type of dataset ('pig_semantic', 'pig_masks', or 'human').

    Returns:
        Extracted image ID.
    """
    if dataset in ["pig_semantic", "pig_masks"]:
        return image_name[5:]
    elif dataset == "human":
        return image_name[13:]
    else:
        raise ValueError(f"Unsupported dataset type: {dataset}")


@array_conversion_decorator
def create_data_summary(
    data: TensorType,
    labels: TensorType | None = None,
    subject_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Create a summary of the loaded data.

    Args:
        data: Main data array.
        labels: Optional label array.
        subject_ids: Optional list of subject IDs.

    Returns:
        Dictionary containing data summary statistics.
    """
    summary = {
        "data_shape": data.shape,
        "data_type": str(data.dtype),
        "data_range": (float(np.min(data)), float(np.max(data))),
        "data_mean": float(np.mean(data)),
        "data_std": float(np.std(data)),
    }

    if labels is not None:
        unique_labels = np.unique(labels)
        summary["unique_labels"] = len(unique_labels)
        summary["label_distribution"] = dict(
            zip(unique_labels, np.bincount(labels), strict=False)
        )

    if subject_ids is not None:
        summary["unique_subjects"] = len(set(subject_ids))
        summary["samples_per_subject"] = len(data) / len(set(subject_ids))

    return summary
