"""
Generate final data specification for human kidney data with both polygon
and semantic annotations.

This script creates a data specification JSON file that maps polygon annotation
data to the appropriate train/val/test splits while avoiding duplicates from
semantic annotations.
"""

import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from htc import DataSpecification, median_table

from mcmlnet.utils.logging import setup_logging

load_dotenv()
logger = setup_logging(level="info", logger_name=__name__)

# Configuration constants
TARGET_DIR = os.getenv("cache_dir")
SPEC_NAME = (
    "human_masks-only_physiological-kidney_5folds_nested-0-2_mapping-12_seed-0.json"
)
DATASET_NAME = "2021_07_26_Tivita_multiorgan_human"
SEMANTIC_SPEC_FILE = (
    "human_semantic-only_physiological-kidney_5folds_nested-0-2_mapping-12_seed-0.json"
)
FOLD_NAME = "fold_0"
DATA_SPLITS = ["train", "val", "test"]
ANNOTATION_NAME = "polygon#annotator1"


def create_empty_fold_spec(fold_name: str) -> dict[str, str | dict[str, list[str]]]:
    """Create an empty fold specification structure.

    Args:
        fold_name: Name of the fold (e.g., "fold_0")

    Returns:
        Dictionary with empty train/val/test splits
    """
    return {
        "fold_name": fold_name,
        "train": {"image_names": []},
        "val": {"image_names": []},
        "test": {"image_names": []},
    }


def get_polygon_image_names(df: pd.DataFrame, annotation_name: str) -> list[str]:
    """Extract polygon image names for given subjects.

    Args:
        df: DataFrame containing annotation data
        annotation_name: Name of the annotation type to filter by

    Returns:
        List of image names with polygon annotations
    """
    return df.query(  # type: ignore [no-any-return]
        f"subject_name in @subject_names and annotation_name == '{annotation_name}'"
    )["image_name"].tolist()


def remove_duplicates_and_semantic_images(
    polygon_image_names: list[str], semantic_image_names: list[str]
) -> list[str]:
    """Remove duplicates and semantic images from polygon image names.

    Args:
        polygon_image_names: List of polygon annotation image names
        semantic_image_names: List of semantic annotation image names

    Returns:
        Cleaned list of polygon image names
    """
    # Remove duplicates
    unique_polygon_names = list(set(polygon_image_names))

    # Remove semantic image names to avoid redundant data
    return [
        img_name
        for img_name in unique_polygon_names
        if img_name not in semantic_image_names
    ]


def process_split_data(
    spec: DataSpecification,
    df: pd.DataFrame,
    fold_name: str,
    split_name: str,
    annotation_name: str,
) -> list[str]:
    """Process data for a specific split and fold.

    Args:
        spec: DataSpecification object
        df: DataFrame containing annotation data
        fold_name: Name of the fold
        split_name: Name of the split (train/val/test)
        annotation_name: Name of the annotation type

    Returns:
        List of image names for the split
    """
    paths = spec.fold_paths(fold_name=fold_name, split_name=split_name)
    subject_names = {p.subject_name for p in paths}

    logger.info(f"Split: {split_name}, Amount of subjects: {len(subject_names)}")
    logger.info(f"Amount of images: {len(paths)}")

    # Get polygon image names for these subjects
    polygon_image_names = df.query(
        f"subject_name in @subject_names and annotation_name == '{annotation_name}'"
    )["image_name"].tolist()
    logger.info(
        f"Amount of images with polygon annotations: {len(polygon_image_names)}"
    )

    # Get semantic image names to avoid redundancy
    semantic_image_names = [p.image_name() for p in paths]

    # Clean the polygon image names
    cleaned_polygon_names = remove_duplicates_and_semantic_images(
        polygon_image_names, semantic_image_names
    )
    logger.info(
        f"Amount of images after duplicate removal: {len(cleaned_polygon_names)}"
    )

    return sorted(cleaned_polygon_names)


def generate_data_specification() -> list[dict[str, str | dict[str, list[str]]]]:
    """Generate the complete data specification.

    Returns:
        List containing fold specifications with train/val/test splits
    """
    spec = DataSpecification(SEMANTIC_SPEC_FILE)
    df = median_table(dataset_name=DATASET_NAME)

    # Process with activated test set to process all data
    data_spec = []
    with spec.activated_test_set():
        # Create fold specification
        fold_specs = create_empty_fold_spec(FOLD_NAME)

        # Process each data split
        for split in DATA_SPLITS:
            image_names = process_split_data(
                spec, df, FOLD_NAME, split, ANNOTATION_NAME
            )
            fold_specs[split]["image_names"] = image_names  # type: ignore [index]

        data_spec.append(fold_specs)

    return data_spec


def save_data_specification(
    data_spec: list[dict[str, str | dict[str, list[str]]]],
    target_dir: str,
    spec_name: str,
) -> None:
    """Save the data specification to a JSON file.

    Args:
        data_spec: Data specification to save
        target_dir: Target directory path
        spec_name: Name of the specification file
    """
    specs_path = Path(target_dir) / spec_name

    with specs_path.open("w") as f:
        json.dump(data_spec, f, indent=4)
        # Add newline at the end for consistent formatting
        f.write("\n")

    logger.info(f"Data specification saved to: {specs_path}")


def main() -> None:
    """Main function to generate and save the data specification."""
    if not TARGET_DIR:
        raise ValueError("Environment variable 'cache_dir' is not set")
    data_spec = generate_data_specification()
    save_data_specification(data_spec, TARGET_DIR, SPEC_NAME)


if __name__ == "__main__":
    main()
