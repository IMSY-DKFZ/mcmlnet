"""Data aggregation functionality."""

import numpy as np
import torch
from rich.progress import track
from torch.utils.data import DataLoader

from mcmlnet.experiments.data_loaders.utils import get_image_id, get_subject_id
from mcmlnet.utils.load_configs import label_is_organic
from mcmlnet.utils.logging import setup_logging

logger = setup_logging(level="info", logger_name=__name__)


def aggregate_data_image_level(
    dataloader: DataLoader,
    dataset: str,
    fit_on_organic_only: bool = True,
) -> dict[str, list | np.ndarray]:
    """Aggregate data at the image level.

    Args:
        dataloader: DataLoader containing the data.
        dataset: Name of the dataset.
        fit_on_organic_only: Whether to filter for organic tissue only.

    Returns:
        Dictionary containing aggregated data.
    """
    if dataloader.batch_size != 1:
        raise ValueError("Batch size must be 1 for image level aggregation")

    mean_spectra = []
    label_ids = []
    subject_ids = []
    image_ids = []

    # Iterate over the images
    for batch in track(
        dataloader,
        total=len(dataloader),
        description="Collecting real data mean spectra",
    ):
        # Load and clamp spectra to physically meaningful values
        img = torch.clamp(batch["features"], 0, 1)
        img = img.view(-1, img.shape[-1]).cpu().numpy()
        labels = batch["labels"].view(-1).cpu().numpy()

        # Extract subject and image IDs based on dataset
        image_name = batch["image_name"][0]
        subject_id = get_subject_id(image_name, dataset)
        image_id = get_image_id(image_name, dataset)

        # Collect average spectra per image
        for label in np.unique(labels):
            label_mask = labels == label
            if np.sum(label_mask) > 0:  # Ensure there are pixels for the label
                mean_spectra.append(np.mean(img[label_mask], axis=0))
                label_ids.append(label)
                subject_ids.append(subject_id)
                image_ids.append(image_id)
            else:
                logger.warning(
                    f"No pixels found for label {label} in image {image_name}"
                )

    # Convert lists to numpy arrays
    mean_spectra_np = np.stack(mean_spectra)
    label_ids_np = np.array(label_ids)
    subject_ids_np = np.array(subject_ids)
    image_ids_np = np.array(image_ids)

    if fit_on_organic_only:
        # Filter out non-organic spectra
        organic_mask = label_is_organic(label_ids_np, dataset)
        mean_spectra_np = mean_spectra_np[organic_mask]
        label_ids_np = label_ids_np[organic_mask]
        subject_ids_np = subject_ids_np[organic_mask]
        image_ids_np = image_ids_np[organic_mask]

    return {
        "mean_spectra": mean_spectra_np,
        "label_ids": label_ids_np.tolist(),
        "subject_ids": subject_ids_np.tolist(),
        "image_ids": image_ids_np.tolist(),
    }


def aggregate_data_subject_level(
    dataloader: DataLoader,
    dataset: str,
    fit_on_organic_only: bool = True,
) -> dict[str, list | np.ndarray]:
    """Aggregate data at the subject level.

    Args:
        dataloader: DataLoader containing the data.
        dataset: Name of the dataset.
        fit_on_organic_only: Whether to filter for organic tissue only.

    Returns:
        Dictionary containing aggregated data.
    """
    if dataloader.batch_size != 1:
        raise ValueError("Batch size must be 1 for subject level aggregation")

    # Collect all subject IDs
    subjects = []
    for batch in track(
        dataloader,
        total=len(dataloader),
        description="Collecting subject IDs",
    ):
        subjects.append(get_subject_id(batch["image_name"][0], dataset))
    unique_subjects = list(set(subjects))

    # Iterate over the unique subjects
    mean_spectra = []
    label_ids = []
    subject_ids = []

    for subject in track(
        unique_subjects,
        total=len(unique_subjects),
        description="Collecting mean spectra per label and subject",
    ):
        spectra: dict[int, list[np.ndarray]] = {}

        for batch in dataloader:
            # Ignore images that do not belong to the current subject
            if get_subject_id(batch["image_name"][0], dataset) != subject:
                continue

            # Load and clamp spectra to physically meaningful values
            img = torch.clamp(batch["features"], 0, 1)
            img = img.view(-1, img.shape[-1]).cpu().numpy()
            labels = batch["labels"].view(-1).cpu().numpy()

            # Collect the spectra sorted by label
            for label in np.unique(labels):
                if label not in spectra:
                    spectra[label] = []
                spectra[label].append(img[labels == label])

        # Compute the mean spectra per label per subject
        for label, data in spectra.items():
            data = np.concatenate(data, axis=0)
            mean_spectra.append(np.mean(data, axis=0))
            label_ids.append(label)
            subject_ids.append(subject)

    # Convert lists to numpy arrays
    mean_spectra_np = np.stack(mean_spectra)
    label_ids_np = np.array(label_ids)
    subject_ids_np = np.array(subject_ids)

    if fit_on_organic_only:
        # Filter out non-organic spectra
        organic_mask = label_is_organic(label_ids_np, dataset)
        mean_spectra_np = mean_spectra_np[organic_mask]
        label_ids_np = label_ids_np[organic_mask]
        subject_ids_np = subject_ids_np[organic_mask]

    return {
        "mean_spectra": mean_spectra_np,
        "label_ids": label_ids_np.tolist(),
        "subject_ids": subject_ids_np.tolist(),
    }
