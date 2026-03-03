"""
Monte Carlo data generation script for inverse functional imaging.

This script generates reflectance datasets for functional spectral imaging
with constant SaO2 across layers, as in standard simulations. It supports
various sampling strategies and can be run on cluster environments.

References:
- https://link.springer.com/chapter/10.1007/978-3-319-66179-7_16/tables/1
- https://link.springer.com/article/10.1007/s11548-016-1376-5/tables/1

Parameter Ranges:
- SaO2: [0, 1] (oxygen saturation)
- vHb: [0, 0.3] (hemoglobin volume fraction)
- a_mie: [500, 5000] (Mie scattering coefficient)
- b_mie: [0.3, 3] (Mie scattering anisotropy)
- a_ray: 0 (Rayleigh scattering coefficient)
- g: [0.8, 0.95] (anisotropy factor)
- n: [1.33, 1.54] (refractive index)
- d: [2e-5, 0.002] m (layer thickness)

"""

import argparse
import os
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    from codecarbon import EmissionsTracker
except ImportError:
    EmissionsTracker = None
from dotenv import load_dotenv
from scipy.stats.qmc import LatinHypercube

from mcmlnet.utils.logging import setup_logging
from mcmlnet.utils.mc_runner import RecomputeMC

load_dotenv()
logger = setup_logging(level="info", logger_name=__name__)

# Global constants
OUTPUT_DIR = os.path.join(
    str(os.getenv("data_dir")), "raw/base_physio_and_physical_simulations", ""
)
N_WVL = 351
IGNORE_A = True
VERBOSE = True
TIMEOUT = None
N_LAYERS = 3

# Type alias for parameter ranges
ParameterRanges = dict[str, list[float]]

# Physiological parameter ranges
PHYSIO_PARAM_RANGES: ParameterRanges = {
    "sao2": [0.001, 1.0],
    "vhb": [0.001, 0.3],
    "a_mie": [500, 5000],
    "b_mie": [0.3, 3.0],
    "a_ray": [0.0, 0.0],  # Fixed at 0
    "g": [0.8, 0.95],
    "n": [1.33, 1.54],
    "d": [2e-5, 0.002],
}


class DummyTracker:
    """Dummy tracker that implements the context manager protocol.

    This class provides a no-op implementation of the EmissionsTracker interface
    when codecarbon is not available. It allows the code to use the same 'with'
    statement syntax regardless of whether emissions tracking is available.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the dummy tracker.

        Args:
            **kwargs: Ignored arguments (for compatibility with EmissionsTracker)
        """
        logger.warning("Using dummy emissions tracker - no emissions will be tracked")

    def __enter__(self) -> "DummyTracker":
        """Enter the context manager.

        Returns:
            self: The tracker instance
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context manager.

        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any)
            exc_tb: Exception traceback (if any)

        Returns:
            bool: False to re-raise exceptions, True to suppress them
        """
        return None  # Don't suppress exceptions


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the toy data generation script.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate Monte Carlo toy datasets for inverse functional imaging"
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
        default=1,
    )
    parser.add_argument(
        "-batch_size",
        "--batch_size",
        help="Number of samples per batch",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "-n_samples",
        "--n_samples",
        help="Total number of samples to generate",
        type=int,
        default=10**6,
    )
    parser.add_argument(
        "-skip_n_samples",
        "--skip_n_samples",
        help="Number of initial samples to skip",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-red_n_wvl",
        "--red_n_wvl",
        help="Number of reduced wavelengths",
        type=int,
        default=15,
    )
    parser.add_argument(
        "-nr_photons",
        "--nr_photons",
        help="Number of photons for Monte Carlo simulation",
        type=int,
        default=10**8,
    )
    parser.add_argument(
        "-data_distribution",
        "--data_distribution",
        help="Data distribution type",
        type=str,
        default="lhs_uniform",
        choices=["uniform", "lhs_uniform", "loguniform", "lhs_loguniform"],
    )

    return parser.parse_args()


def generate_physiological_parameters(
    n_samples: int,
    distribution: str = "lhs_uniform",
    seed: int = 42,
    n_layers: int = N_LAYERS,
) -> np.ndarray:
    """Generate physiological dataset parameters using specified distribution.

    Args:
        n_samples: Number of parameter sets to generate
        distribution: Distribution type for sampling
        seed: Random seed for reproducibility
        n_layers: Number of tissue layers

    Returns:
        np.ndarray: Generated parameters with shape (n_samples, n_layers * n_params)

    Raises:
        ValueError: If distribution is not supported or reference file not found
    """
    np.random.seed(seed)

    # Number of parameters per layer
    n_params_per_layer = len(PHYSIO_PARAM_RANGES)
    total_params = n_layers * n_params_per_layer

    if distribution == "uniform":
        # Uniform sampling
        params = np.zeros((n_samples, total_params))
        for i, (_param_name, param_range) in enumerate(PHYSIO_PARAM_RANGES.items()):
            min_val, max_val = param_range[0], param_range[1]
            for layer in range(n_layers):
                col_idx = layer * n_params_per_layer + i
                params[:, col_idx] = np.random.uniform(min_val, max_val, n_samples)

    elif distribution == "lhs_uniform":
        # Latin Hypercube Sampling with uniform distribution
        lhs = LatinHypercube(d=total_params, seed=seed)
        samples = lhs.random(n=n_samples)

        params = np.zeros((n_samples, total_params))
        for i, (_param_name, param_range) in enumerate(PHYSIO_PARAM_RANGES.items()):
            min_val, max_val = param_range[0], param_range[1]
            for layer in range(n_layers):
                col_idx = layer * n_params_per_layer + i
                params[:, col_idx] = samples[:, col_idx] * (max_val - min_val) + min_val

    elif distribution == "loguniform":
        # Log-uniform sampling for appropriate parameters
        params = np.zeros((n_samples, total_params))
        for i, (param_name, param_range) in enumerate(PHYSIO_PARAM_RANGES.items()):
            min_val, max_val = param_range[0], param_range[1]
            for layer in range(n_layers):
                col_idx = layer * n_params_per_layer + i
                if param_name in ["a_mie", "d"] and min_val > 0:
                    # Log-uniform for positive parameters
                    log_min, log_max = np.log(min_val), np.log(max_val)
                    params[:, col_idx] = np.exp(
                        np.random.uniform(log_min, log_max, n_samples)
                    )
                else:
                    # Uniform for other parameters
                    params[:, col_idx] = np.random.uniform(min_val, max_val, n_samples)

    elif distribution == "lhs_loguniform":
        # Latin Hypercube Sampling with log-uniform distribution
        lhs = LatinHypercube(d=total_params, seed=seed)
        samples = lhs.random(n=n_samples)

        params = np.zeros((n_samples, total_params))
        for i, (param_name, param_range) in enumerate(PHYSIO_PARAM_RANGES.items()):
            min_val, max_val = param_range[0], param_range[1]
            for layer in range(n_layers):
                col_idx = layer * n_params_per_layer + i
                if param_name in ["a_mie", "d"] and min_val > 0:
                    # Log-uniform for positive parameters
                    log_min, log_max = np.log(min_val), np.log(max_val)
                    params[:, col_idx] = np.exp(
                        samples[:, col_idx] * (log_max - log_min) + log_min
                    )
                else:
                    # Uniform for other parameters
                    params[:, col_idx] = (
                        samples[:, col_idx] * (max_val - min_val) + min_val
                    )
    else:
        # Load parameter data from file
        reference = os.path.join(
            os.environ["data_dir"],
            "raw/base_physio_and_physical_simulations",
            "reference_parameters.csv",
        )
        if os.path.isfile(reference):
            # Load parameter data from file and run sanity parameter value checks
            params = pd.read_csv(reference, header=[0, 1])
            params = params.to_numpy()[:, :24]

            for param_id, (name, param_range) in enumerate(PHYSIO_PARAM_RANGES.items()):
                all_layer_param_ids = [param_id + i * 8 for i in range(3)]
                if not np.all(params[:, all_layer_param_ids]) >= param_range[0]:
                    raise AssertionError(f"Too small {name} values detected!")
                if not np.all(params[:, all_layer_param_ids]) <= param_range[1]:
                    raise AssertionError(f"Too large {name} values detected!")

            if len(params) != n_samples:
                raise ValueError(
                    f"Reference file has {len(params)} samples, "
                    f"but {n_samples} requested"
                )
        else:
            raise ValueError(
                f"Reference file {reference} not found or method not implemented. "
                "Please provide a valid reference file."
            )

    if params.shape[1] % 9 != 0:
        # Insert default cHb columns if not already present
        cHb = np.ones((n_samples,)) * 150.0
        params = np.insert(params, 8, cHb, axis=1)
        params = np.insert(params, 17, cHb, axis=1)
        params = np.insert(params, 26, cHb, axis=1)

    # Set deepest layer to always be 100x the thickest value (usually 0.2 m)
    params[:, -2] = PHYSIO_PARAM_RANGES["d"][1] * 100

    # Log parameter statistics
    logger.info(f"Parameter array shape: {params.shape}")
    for i, param_name in enumerate([*PHYSIO_PARAM_RANGES.keys(), "cHb"]):
        for layer in range(n_layers):
            col_idx = layer * (n_params_per_layer + 1) + i
            col = params[:, col_idx]
            logger.info(
                f"{param_name} (layer {layer+1}): "
                f"min={col.min():.4g}, max={col.max():.4g}, "
                f"mean={col.mean():.4g}, std={col.std():.4g}"
            )

    return params


def _generate_folder_name(n_samples: int, red_n_wvl: int, nr_photons: int) -> str:
    """Generate folder name for the simulation output.

    Args:
        n_samples: Total number of samples
        red_n_wvl: Number of reduced wavelengths
        nr_photons: Number of photons

    Returns:
        str: Generated folder name
    """
    return (
        f"three_layer_absorption_{n_samples // 1000}k"
        f"_{red_n_wvl}_wvl_{nr_photons // 10**6}"
        f"_mio_photons_8_point_precision_thick/"
    )


def _configure_batch_processing(
    folder_name: str,
    n_runs: int,
    run_id: int | None,
    batch_size: int,
    n_samples: int,
    skip_n_samples: int,
    red_n_wvl: int,
) -> tuple[str, tuple[int, int]]:
    """Configure batch processing parameters.

    Args:
        folder_name: Base folder name
        n_runs: Total number of (separate) runs
        run_id: Run identifier
        skip_n_samples: Number of samples to skip
        n_samples: Total number of samples
        batch_size: Number of samples per batch
        red_n_wvl: Number of reduced wavelengths (for file naming)

    Returns:
        tuple: (mco_folder, batch_range)

    Raises:
        ValueError: If batch configuration is invalid
    """
    if run_id is None or n_runs == 1:
        # Single batch processing
        mco_folder = os.path.join(OUTPUT_DIR, folder_name, f"mco_{red_n_wvl}_wvl_all/")
        batch_range = (0, -1)
    else:
        # Multi-batch processing
        mco_folder = os.path.join(
            OUTPUT_DIR, folder_name, f"mco_{red_n_wvl}_wvl_{run_id + 1}/"
        )

        # Calculate batch bounds
        skip_batches = skip_n_samples // batch_size
        effective_batch_size = (n_samples - skip_n_samples) // (batch_size * n_runs)

        lower_batch_bound = effective_batch_size * run_id + skip_batches
        upper_batch_bound = effective_batch_size * (run_id + 1) + skip_batches

        batch_range = (lower_batch_bound, upper_batch_bound)

        logger.info(
            f"Run {run_id}: skip_batches={skip_batches}, "
            f"bounds=({lower_batch_bound}, {upper_batch_bound})"
        )

    return mco_folder, batch_range


def run_monte_carlo_simulations(
    parameters: np.ndarray,
    n_runs: int,
    batch_size: int,
    skip_n_samples: int,
    n_samples: int,
    nr_photons: int,
    red_n_wvl: int,
    run_id: int | None = None,
) -> None:
    """Run Monte Carlo simulations for toy dataset parameters.

    Args:
        parameters: Parameter sets for simulation
        n_runs: Total number of (separate) runs
        batch_size: Number of samples per simulation batch
        skip_n_samples: Number of samples to skip
        n_samples: Total number of samples
        nr_photons: Number of photons for simulation
        red_n_wvl: Number of reduced wavelengths
        run_id: Run identifier for file naming

    Raises:
        ValueError: If skip_n_samples is invalid or batch configuration is incorrect
    """
    logger.info(
        f"Starting Monte Carlo simulations for {len(parameters)} parameter sets"
    )

    # Validate skip_n_samples
    if skip_n_samples < 0 or skip_n_samples >= n_samples:
        raise ValueError(
            f"skip_n_samples ({skip_n_samples}) must be greater than 0 and "
            f"smaller than n_samples ({n_samples})"
        )

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize Monte Carlo reflectance simulation wavelengths
    base_wavelengths = np.linspace(300, 1000, N_WVL) * 1e-9
    reduced_wvl = base_wavelengths[:: N_WVL // red_n_wvl][:red_n_wvl]
    logger.info(f"Reduced wavelengths ({red_n_wvl}): {reduced_wvl}")

    # Generate file name and batch configuration
    folder_name = _generate_folder_name(n_samples, red_n_wvl, nr_photons)
    mco_folder, batch_range = _configure_batch_processing(
        folder_name,
        n_runs,
        run_id,
        batch_size,
        n_samples,
        skip_n_samples,
        red_n_wvl,
    )

    # Create MCO folder
    os.makedirs(mco_folder, exist_ok=True)

    # Initialize Monte Carlo simulator
    mc_simulator = RecomputeMC(
        wavelengths=reduced_wvl,
        nr_photons=nr_photons,
        ignore_a=IGNORE_A,
        verbose=VERBOSE,
        timeout=TIMEOUT,
    )

    # Generate column headers and run simulation
    columns = [*PHYSIO_PARAM_RANGES.keys(), "chb"]
    header = mc_simulator.generate_column_headers(columns, N_LAYERS)

    logger.info("Parameter preview:")
    logger.info(pd.DataFrame(parameters, columns=header).head())

    # Run simulation
    mc_simulator.run_simulation_from_df(
        pd.DataFrame(parameters, columns=header),
        save_dir=os.path.join(OUTPUT_DIR, folder_name),
        batch_size=batch_size,
        batch_range=batch_range,
    )


def main() -> None:
    """Main function to orchestrate the data generation process."""
    args = parse_arguments()

    logger.info(f"Starting data generation of run {args.run_id} of {args.n_runs}.")
    logger.info(f"Distribution: {args.data_distribution}")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # Start tracking emissions if codecarbon is installed
    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
        print(f"Running on GPU ID: {gpu_id}")
    else:
        gpu_id = None
        print("No GPU available. Running on CPU.")

    # Create tracker (either real or dummy)
    if EmissionsTracker is not None:
        tracker = EmissionsTracker(
            project_name="MCML sim.s",
            output_dir=OUTPUT_DIR,
            output_file=f"emissions_{args.n_samples}_{args.run_id}.csv",
            gpu_ids=[gpu_id] if gpu_id is not None else [],
            experiment_id=args.run_id,
            measure_power_secs=60,  # measure power every 60 seconds
        )
    else:
        tracker = DummyTracker()
        logger.warning("Codecarbon not installed. Emissions tracking disabled.")

    # Start emissions tracking
    with tracker:
        # Generate parameters
        parameters = generate_physiological_parameters(
            n_samples=args.n_samples,
            distribution=args.data_distribution,
            seed=42,
            n_layers=N_LAYERS,
        )

        # Run Monte Carlo simulations
        run_monte_carlo_simulations(
            parameters=parameters,
            n_runs=args.n_runs,
            batch_size=args.batch_size,
            skip_n_samples=args.skip_n_samples,
            n_samples=args.n_samples,
            nr_photons=args.nr_photons,
            red_n_wvl=args.red_n_wvl,
            run_id=args.run_id,
        )


if __name__ == "__main__":
    main()
