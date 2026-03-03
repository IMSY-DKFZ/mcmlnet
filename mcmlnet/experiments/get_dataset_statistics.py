"""
Collect subject amounts for the splits used during evaluations.

This script analyzes data splits to collect statistics about:
- Subject amounts per label
- Camera IDs used
- Pixel amounts per label
- Pixel amounts per subject and label
"""

from typing import Any

import numpy as np
import pandas as pd
from htc import DataPath
from rich.progress import track

from mcmlnet.experiments.data_loaders.real_data import (
    CombinedHumanDataLoader,
    HumanDataLoader,
    PigDataLoader,
    RealDataLoader,
)
from mcmlnet.utils.load_configs import (
    label_is_organic,
    load_config,
    load_label_map,
)


def load_dataloader(cfg: dict[str, Any]) -> Any:
    """
    Load the appropriate dataloader based on configuration.

    Args:
        cfg: Configuration dictionary

    Returns:
        Dataloader object
    """
    data_loader: RealDataLoader
    if cfg["data"]["test_data"] == "final":
        data_loader = CombinedHumanDataLoader()
    elif cfg["data"]["real_data"] == "human":
        data_loader = HumanDataLoader()
    elif cfg["data"]["real_data"] == "pig":
        data_loader = PigDataLoader()
    else:
        raise ValueError(f"Invalid real data type: {cfg['data']['real_data']}")

    if cfg["data"]["test_data"] == "test":
        return data_loader.load_test_data()
    elif cfg["data"]["test_data"] == "final":
        return data_loader.load_final_data()
    else:
        dataloader, _ = data_loader.load_training_data()
        return dataloader


def extract_batch_metadata(
    batch: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """
    Extract metadata from a batch.

    Args:
        batch: Batch dictionary containing labels and image_name

    Returns:
        Tuple of (unique_labels, counts, subject, camera_id)
    """
    labels = batch["labels"].view(-1).cpu().numpy()
    unique_labels, counts = np.unique(labels, return_counts=True)
    subject = batch["image_name"][0][6:12]
    datapath = DataPath.from_image_name(batch["image_name"][0])
    camera_id = datapath.meta("Camera_CamID")

    return unique_labels, counts, subject, camera_id


def collect_statistics(
    dataloader: Any,
) -> tuple[dict[str, set[int]], set[str], dict[int, int], dict[str, dict[int, int]]]:
    """
    Collect statistics from the dataloader.

    Args:
        dataloader: DataLoader object

    Returns:
        Tuple of (subject_amounts, camera_ids, pixel_amounts, pixel_amounts_per_subject)
    """
    subject_amounts: dict[str, set[int]] = {}
    camera_ids: set[str] = set()
    pixel_amounts: dict[int, int] = {}
    pixel_amounts_per_subject: dict[str, dict[int, int]] = {}

    for batch in track(dataloader, description="Processing batches"):
        unique_labels, counts, subject, camera_id = extract_batch_metadata(batch)

        # Collect unique subjects and their labels
        if subject not in subject_amounts:
            subject_amounts[subject] = set()
        subject_amounts[subject].update(unique_labels)

        # Collect camera IDs
        camera_ids.add(camera_id)

        # Collect pixel amounts per label
        for label, count in zip(unique_labels, counts, strict=False):
            pixel_amounts[label] = pixel_amounts.get(label, 0) + count

        # Collect pixel amounts per label and subject
        if subject not in pixel_amounts_per_subject:
            pixel_amounts_per_subject[subject] = {}
        for label, count in zip(unique_labels, counts, strict=False):
            if label not in pixel_amounts_per_subject[subject]:
                pixel_amounts_per_subject[subject][label] = 0
            pixel_amounts_per_subject[subject][label] += count

    return subject_amounts, camera_ids, pixel_amounts, pixel_amounts_per_subject


def print_statistics(
    subject_amounts: dict[str, set[int]],
    camera_ids: set[str],
    pixel_amounts: dict[int, int],
    label_map: dict[int, str],
    cfg: dict[str, Any],
) -> None:
    """
    Print collected statistics.

    Args:
        subject_amounts: Dictionary mapping subjects to their labels
        camera_ids: Set of camera IDs
        pixel_amounts: Dictionary mapping labels to pixel counts
        label_map: Dictionary mapping label IDs to names
        cfg: Configuration dictionary
    """
    print("Subject amounts per label:")
    for subject, labels in subject_amounts.items():
        label_names = [label_map[label] for label in labels]
        print(f"{subject}: {label_names}")

    print(f"\nCamera IDs ({len(camera_ids)} total):")
    print(sorted(camera_ids))

    print("\nPixel amounts per label:")
    organic_counter = 0
    for label, count in sorted(pixel_amounts.items()):
        label_name = label_map.get(label, f"Unknown_{label}")
        print(f"Label {label_name}: {count:,} pixels")

        if np.any(label_is_organic([label], cfg["data"]["real_data"])):
            organic_counter += count

    print(f"Label Organic: {organic_counter:,} pixels")
    print(f"Total Pixels: {sum(pixel_amounts.values()):,}")


def create_subject_dataframe(
    pixel_amounts_per_subject: dict[str, dict[int, int]], label_map: dict[int, str]
) -> pd.DataFrame:
    """
    Create a DataFrame from pixel amounts per subject.

    Args:
        pixel_amounts_per_subject: Dictionary mapping subjects to label pixel counts
        label_map: Dictionary mapping label IDs to names

    Returns:
        DataFrame with subject statistics
    """
    df = (
        pd.DataFrame.from_dict(pixel_amounts_per_subject, orient="index")
        .fillna(0)
        .astype(int)
    )

    # Map column names to label names
    df.columns = [label_map.get(label, f"Unknown_{label}") for label in df.columns]
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Subject"}, inplace=True)

    return df


def main() -> None:
    """Main function to collect and display data statistics."""
    # Load configuration and label map
    cfg = load_config()
    label_map = load_label_map(dataset=cfg["data"]["real_data"])

    # Load dataloader
    dataloader = load_dataloader(cfg)

    # Collect statistics
    subject_amounts, camera_ids, pixel_amounts, pixel_amounts_per_subject = (
        collect_statistics(dataloader)
    )

    # Print statistics
    print_statistics(subject_amounts, camera_ids, pixel_amounts, label_map, cfg)

    # Create and save DataFrame
    df = create_subject_dataframe(pixel_amounts_per_subject, label_map)
    output_file = "spectra_counts_per_subject.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSubject statistics saved to: {output_file}")
    print(f"\nDataFrame shape: {df.shape}")
    print(df.head())


if __name__ == "__main__":
    main()
