"""
Validation and test data generation script with 1000M photons
(10x regular photon count).

This script generates validation and test datasets with increased photon counts
for more accurate Monte Carlo simulations. It supports both physiological and
physical parameter types and can be run on cluster environments.

"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from rich.progress import track

from mcmlnet.susi.calculate_spectra import (
    calculate_spectrum_for_batch,
    calculate_spectrum_for_physical_batch,
)
from mcmlnet.training.data_loading.preprocessing import PreProcessor
from mcmlnet.utils.loading import SimulationDataLoader
from mcmlnet.utils.logging import setup_logging
from mcmlnet.utils.mc_runner import RecomputeMC

logger = setup_logging(level="info", logger_name=__name__)

# Global constants
OUTPUT_DIR = os.path.join(
    os.environ["data_dir"], "raw/base_physio_and_physical_simulations/", ""
)
N_WVL_PHYSIO = 15
N_WVL_PHYSICAL = 351
N_LAYERS = 3
NR_PHOTONS = 10**9
VERBOSE = True
IGNORE_A = True


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the data generation script.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate validation and test datasets with 1000M photons "
            "(10x regular photon count)"
        )
    )
    parser.add_argument(
        "-run_id",
        "--run_id",
        help="Run ID for separate run processing",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-n_runs",
        "--n_runs",
        help="Total number of separate runs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "-n_batches",
        "--n_batches",
        help="Number of batches to generate",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "-data_type",
        "--data_type",
        help="Data type (physio or physical)",
        type=str,
        default="physio",
    )
    parser.add_argument(
        "-data_split",
        "--data_split",
        help="Data split (val or test)",
        type=str,
        default="val",
    )

    return parser.parse_args()


def validate_arguments(
    args: argparse.Namespace, dataset_size: int
) -> tuple[int, int, int]:
    """Validate command line arguments.

    Args:
        args: Parsed command line arguments
        dataset_size: Size of the dataset

    Returns:
        batch_size: Size of each batch
        batch_min: Minimum batch index
        batch_max: Maximum batch index

    Raises:
        ValueError: If arguments are not valid.
    """
    if args.run_id is None or args.n_runs is None:
        raise ValueError("Run ID and number of runs must be provided.")
    if args.n_batches is None:
        raise ValueError("Number of batches must be provided.")
    if args.data_type not in ["physio", "physical"]:
        raise ValueError("Data type must be either 'physio' or 'physical'.")
    if args.data_split not in ["val", "test"]:
        raise ValueError("Data split must be either 'val' or 'test'.")
    if args.run_id < 0 or args.run_id >= args.n_runs:
        raise ValueError("Run ID must be between 0 and number of runs - 1.")
    if args.n_batches % args.n_runs != 0:
        raise ValueError("Number of batches must be divisible by number of runs.")
    if dataset_size % args.n_batches != 0:
        raise ValueError("Dataset size must be divisible by number of batches.")

    batch_size = dataset_size // args.n_batches
    batch_min = args.run_id * args.n_batches // args.n_runs
    batch_max = (args.run_id + 1) * args.n_batches // args.n_runs

    return batch_size, batch_min, batch_max


def load_data(data_type: str, data_split: str) -> torch.Tensor:
    """Load data based on type and split.

    Args:
        data_type: Type of data ('physio' or 'physical')
        data_split: Data split ('val' or 'test')

    Returns:
        torch.Tensor: Loaded data tensor

    Raises:
        ValueError: If data type or split is not supported.
    """
    if data_type not in ["physio", "physical"]:
        raise ValueError(
            f"Data type '{data_type}' not supported. Use 'physio' or 'physical'."
        )

    if data_split not in ["val", "test"]:
        raise ValueError(
            f"Data split '{data_split}' not supported. Use 'val' or 'test'."
        )

    data_loader = SimulationDataLoader()

    if data_type == "physio":
        # Load physiological data
        data = data_loader.load_simulation_data(
            "raw/base_physio_and_physical_simulations/physiological_training_100M_photons_1M_samples.parquet",
        )
    else:
        # Load physical data
        data = data_loader.load_physical_simulation_data(
            "raw/base_physio_and_physical_simulations/physical_generalization_100M_photons.parquet",
            n_wavelengths=N_WVL_PHYSICAL,
        )

    # Apply consistent split
    split_data = data[PreProcessor().consistent_data_split_ids(data, data_split)]

    logger.info(f"Loaded {data_type} {data_split} data: {split_data.shape}")
    return split_data


def _compute_batch(
    base_data: torch.Tensor,
    batch_idx: int,
    batch_size: int,
    data_type: str,
    header: pd.Index | pd.MultiIndex,
    wavelengths: np.ndarray,
    run_id: int,
) -> pd.DataFrame:
    """Compute a single batch of data.

    Args:
        base_data: Base data tensor
        batch_idx: Current batch index
        batch_size: Size of each batch
        data_type: Type of data ('physio' or 'physical')
        header: Column headers for the DataFrame
        wavelengths: Wavelength array
        run_id: Current run ID

    Returns:
        DataFrame with computed results
    """
    start_idx = batch_idx * batch_size
    end_idx = (batch_idx + 1) * batch_size

    if data_type == "physio":
        data_batch = base_data[start_idx:end_idx, : N_LAYERS * 8]
        return calculate_spectrum_for_batch(
            pd.DataFrame(data_batch, columns=header),
            wavelengths,
            nr_photons=NR_PHOTONS,
            mci_base_folder=f"mci_{batch_idx}_{run_id}",
            batch_id=str(batch_idx),
            ignore_a=IGNORE_A,
            mco_file=f"batch_{run_id}.mco",
            verbose=VERBOSE,
        )
    else:  # physical
        data_batch = base_data[start_idx:end_idx, :, : N_LAYERS * 5].copy()
        # Reshape into 2D format and convert to simulation-compatible DataFrame
        data_batch = data_batch.reshape(-1, N_LAYERS * 5)
        physical_order = [
            layer * N_LAYERS + layer_id
            for layer_id in range(N_LAYERS)
            for layer in range(5)
        ]
        data_batch = data_batch[:, physical_order]
        result_df = pd.DataFrame(data_batch, columns=header)

        # Compute reflectance
        reflectance = calculate_spectrum_for_physical_batch(
            batch=result_df,
            wavelengths=np.array(
                [400 * 10**-9]
            ),  # dummy wavelength, as scattering/absorption already computed
            nr_photons=NR_PHOTONS,
            mci_base_folder=f"mci_{batch_idx}_{run_id}",
            batch_id=str(batch_idx),
            ignore_a=IGNORE_A,
            mco_file=f"batch_{run_id}.mco",
            verbose=VERBOSE,
        )
        # Add reflectance to result DataFrame
        result_df[("reflectance", "", "")] = reflectance.reflectances.to_numpy()

        return result_df


def main() -> None:
    """Main function to orchestrate the 1000M photon data generation."""
    args = parse_arguments()

    logger.info("Starting 1000M photon data generation")
    logger.info(f"Data type: {args.data_type}, Split: {args.data_split}")
    logger.info(f"Run {args.run_id} of {args.n_runs}")

    # Load base data which is to be resimulated
    base_data = load_data(args.data_type, args.data_split)

    # Validate arguments
    batch_size, batch_min, batch_max = validate_arguments(args, len(base_data))

    # Setup wavelengths based on data type
    wavelengths = (
        np.linspace(300, 1000, num=N_WVL_PHYSICAL, endpoint=True, dtype=float) * 10**-9
    )
    if args.data_type == "physio":
        wavelengths = wavelengths[:: N_WVL_PHYSICAL // N_WVL_PHYSIO][:N_WVL_PHYSIO]
    print(f"Using {len(wavelengths)} wavelengths for simulation.")

    # Setup column headers based on data type
    if args.data_type == "physio":
        columns = ["sao2", "vhb", "a_mie", "b_mie", "a_ray", "g", "n", "d"]
        header = RecomputeMC.generate_column_headers(columns, N_LAYERS)
    else:  # physical
        dummy_wvl = 400 * 10**-9
        header = pd.MultiIndex.from_product(
            [
                [f"layer{i}" for i in range(N_LAYERS)],
                [np.round(dummy_wvl, 12)],
                ["ua", "us", "g", "n", "d"],
            ],
            names=["layer [top first]", "wavelength [m]", "parameter"],
        )

    # Setup result directory
    result_dir = os.path.join(
        OUTPUT_DIR,
        (
            f"{args.data_type}_training_100M_1M_samples_{args.data_split}_10x_photons"
            if args.data_type == "physio"
            else f"physical_generalization_100M_{args.data_split}_10x_photons"
        ),
    )
    os.makedirs(result_dir, exist_ok=True)

    # Process batches
    for batch_idx in track(
        range(args.n_batches), description="Batch Processing", total=args.n_batches
    ):
        # Skip batches not assigned to this run
        if batch_idx < batch_min or batch_idx >= batch_max:
            continue

        batch_file = os.path.join(result_dir, f"batch_{batch_idx}.csv")

        # Try to load existing batch, otherwise compute it
        try:
            header_rows = [0, 1] if args.data_type == "physio" else [0, 1, 2]
            result_df = pd.read_csv(batch_file, header=header_rows)
        except FileNotFoundError:
            result_df = _compute_batch(
                base_data,
                batch_idx,
                batch_size,
                args.data_type,
                header,
                wavelengths,
                args.run_id,
            )
            result_df.to_csv(batch_file, index=False)


if __name__ == "__main__":
    main()
