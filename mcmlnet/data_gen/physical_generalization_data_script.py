"""
Monte Carlo physical data generation script to test surrogate model generalization.

This script generates physical parameter datasets for forward surrogate model
evaluation on unseen distributions, using a hand-crafted sampling strategy
for a three-layer tissue model.

Sampling Strategies:
- Three layer model with hand-crafted physical parameter sampling

Parameter Ranges:
- mu_a (absorption coefficient): [0.001-2*10^6] m^-1
- mu_s (scattering coefficient): [0.1-2*10^6] m^-1
- g (anisotropy factor): [0.8-0.95]
- n (refractive index): [1.33-1.54]
- d (layer thickness): [2e-5-0.002] m

"""

import argparse
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rich.progress import track
from scipy.stats.qmc import LatinHypercube

from mcmlnet.susi.calculate_spectra import calculate_spectrum_for_physical_batch
from mcmlnet.utils.logging import setup_logging

load_dotenv()
logger = setup_logging(level="info", logger_name=__name__)

# Global constants
OUTPUT_DIR = os.path.join(
    os.environ["data_dir"], "raw/base_physio_and_physical_simulations/", ""
)
IGNORE_A = True
VERBOSE = True
N_WVL = 351
NR_PHOTONS = 10**8

# Physical parameter definitions
N_LAYERS = 3
N_PARAMS = 5
MU_A_MIN, MU_A_MAX = 0.1, 2 * 10**5  # [m^-1]
MU_S_MIN, MU_S_MAX = 10, 5 * 10**5  # [m^-1]
G_MIN, G_MAX = 0.8, 0.95
N_MIN, N_MAX = 1.33, 1.54
D_MIN, D_MAX = 1e-5, 0.002  # [m]


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the data generation script.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate Monte Carlo physical parameter datasets "
            "for surrogate model training"
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
        default=10**5 * N_WVL,
    )
    parser.add_argument(
        "-nr_photons",
        "--nr_photons",
        help="Number of photons for Monte Carlo simulation",
        type=int,
        default=10**8,
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments.

    Args:
        args: Parsed command line arguments

    Raises:
        ValueError: If arguments are invalid
    """
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    if args.n_runs <= 0:
        raise ValueError(f"n_runs must be positive, got {args.n_runs}")
    if args.n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {args.n_samples}")
    if args.nr_photons <= 0:
        raise ValueError(f"nr_photons must be positive, got {args.nr_photons}")
    if args.run_id is not None and (args.run_id < 0 or args.run_id >= args.n_runs):
        raise ValueError(
            f"run_id must be between 0 and {args.n_runs - 1}, got {args.run_id}"
        )


def get_consistent_parameter_batch(
    data: np.ndarray,
    run_id: int | None = None,
    n_runs: int = 1,
) -> np.ndarray:
    """Extract a consistent batch of parameters from the full dataset.

    Args:
        data: Full parameter dataset
        run_id: Run identifier for consistent sampling
        n_runs: Total number of runs

    Returns:
        np.ndarray: Batch of parameters for simulation

    Raises:
        ValueError: If run_id is invalid
    """
    if run_id is None:
        return data

    if run_id < 0 or run_id >= n_runs:
        raise ValueError(f"run_id must be between 0 and {n_runs - 1}, got {run_id}")

    # Calculate batch size and offset
    batch_size = len(data) // n_runs
    start_idx = run_id * batch_size
    end_idx = start_idx + batch_size

    batch_data = data[start_idx:end_idx]
    logger.info(f"Extracted run {run_id}: {len(batch_data)} samples")

    return batch_data


def validate_parameter_ranges() -> None:
    """Validate that parameter ranges are valid.

    Raises:
        ValueError: If parameter ranges are invalid
    """
    if MU_A_MIN <= 0 or MU_S_MIN <= 0:
        raise ValueError("mu_a and mu_s must be > 0")
    if MU_A_MAX <= MU_A_MIN:
        raise ValueError(f"MU_A_MAX ({MU_A_MAX}) must be > MU_A_MIN ({MU_A_MIN})")
    if MU_S_MAX <= MU_S_MIN:
        raise ValueError(f"MU_S_MAX ({MU_S_MAX}) must be > MU_S_MIN ({MU_S_MIN})")
    if G_MAX <= G_MIN:
        raise ValueError(f"G_MAX ({G_MAX}) must be > G_MIN ({G_MIN})")
    if N_MAX <= N_MIN:
        raise ValueError(f"N_MAX ({N_MAX}) must be > N_MIN ({N_MIN})")
    if D_MAX <= D_MIN:
        raise ValueError(f"D_MAX ({D_MAX}) must be > D_MIN ({D_MIN})")


def reorder_parameters(parameter_array: np.ndarray, n_layers: int) -> np.ndarray:
    """
    Reorder parameters to match the simulation DataFrame header
    from grouped n_layer x mu_a, n_layer x us, ... to layer-wise
    mu_a, us, g, n, d, mu_a, us, g, n, d, ... ordering.

    Args:
        parameter_array: Array of physical tissue parameters
        n_layers: Number of tissue layers

    Returns:
        Reordered parameters
    """
    col_ids = np.array([i + j * n_layers for i in range(n_layers) for j in range(5)])
    return parameter_array[:, col_ids]


def generate_simulation_header(
    n_layers: int, wavelengths: float | np.ndarray
) -> pd.MultiIndex:
    """Generate a multi-index header for a simulation DataFrame.

    Args:
        n_layers: Number of tissue layers
        wavelengths: Wavelengths for DataFrame header

    Returns:
        Multi-index header for a simulation DataFrame
    """
    rounded_to_picometer_precision = np.round(wavelengths, 12)
    if isinstance(wavelengths, float):
        rounded_to_picometer_precision = [rounded_to_picometer_precision]
    else:
        rounded_to_picometer_precision = rounded_to_picometer_precision.tolist()
    return pd.MultiIndex.from_product(
        [
            [f"layer{i}" for i in range(n_layers)],
            rounded_to_picometer_precision,
            ["ua", "us", "g", "n", "d"],
        ],
        names=["layer [top first]", "wavelength [m]", "parameter"],
    )


def hand_crafted_layer_generation(
    n_samples: int,
    n_wvl: int,
    n_layers: int,
    infinite_thickness: bool = True,
    seed: int = 42,
) -> np.ndarray:
    """Generate hand-crafted physical parameters using LHS for most relevant parameters.

    This function generates parameters using Latin Hypercube Sampling for the 10 most
    relevant parameters (based on preliminary Shapley analysis) and fills the remaining
    5 parameters with uniform random values.

    Most relevant parameters (10):
    - mu_a_1, mu_a_2, mu_a_3, mu_s_1, mu_s_2, mu_s_3, g_1, g_2, d_1, n_1
    Less important parameters (5):
    - g_3, d_2, d_3, n_2, n_3

    NOTE:
    Going beyond 100000*351 samples is not recommended, as the latin hypercube
    sampling becomes very slow quickly with growing sample amount.

    Args:
        n_samples: Number of parameter sets to generate
        n_wvl: Number of wavelengths
        n_layers: Number of tissue layers
        infinite_thickness: Whether to use infinite thickness assumption
        seed: Random seed for reproducibility

    Returns:
        Generated parameters with shape (n_samples, n_pseudo_wvl, n_layers * n_params)

    Raises:
        ValueError: If parameter ranges are invalid or output shape is incorrect
    """
    # Validate parameter ranges
    validate_parameter_ranges()

    # Initialize parameter array
    tissue_params = np.zeros((n_samples, N_LAYERS * N_PARAMS))

    # Generate LHS samples for the 10 most relevant parameters
    lhs = LatinHypercube(d=10, seed=seed)
    lhs_samples = lhs.random(n=n_samples)

    # Rescale to sqrt-space ("softer" than log-space)
    sqrt_mu_a_min, sqrt_mu_a_max = np.sqrt(MU_A_MIN), np.sqrt(MU_A_MAX)
    sqrt_mu_s_min, sqrt_mu_s_max = np.sqrt(MU_S_MIN), np.sqrt(MU_S_MAX)
    sqrt_d_min, sqrt_d_max = np.sqrt(D_MIN), np.sqrt(D_MAX)

    # Fill absorption coefficients (mu_a) - layers 0, 1, 2
    tissue_params[:, :N_LAYERS] = (
        lhs_samples[:, :N_LAYERS] * (sqrt_mu_a_max - sqrt_mu_a_min) + sqrt_mu_a_min
    ) ** 2

    # Fill scattering coefficients (mu_s) - layers 0, 1, 2
    tissue_params[:, N_LAYERS : 2 * N_LAYERS] = (
        lhs_samples[:, N_LAYERS : 2 * N_LAYERS] * (sqrt_mu_s_max - sqrt_mu_s_min)
        + sqrt_mu_s_min
    ) ** 2

    # Fill anisotropy factors (g) - layers 0, 1 (most relevant)
    tissue_params[:, 2 * N_LAYERS : 2 * N_LAYERS + 2] = (
        lhs_samples[:, 6:8] * (G_MAX - G_MIN) + G_MIN
    )

    # Fill refractive index (n) - layer 0 (most relevant)
    tissue_params[:, 3 * N_LAYERS] = lhs_samples[:, 8] * (N_MAX - N_MIN) + N_MIN

    # Fill layer thickness (d) - layer 0 (most relevant)
    tissue_params[:, 4 * N_LAYERS] = (
        lhs_samples[:, 9] * (sqrt_d_max - sqrt_d_min) + sqrt_d_min
    ) ** 2

    # Fill remaining parameters with random uniform values
    random_state = np.random.RandomState(seed=seed)

    # g_3 (layer 2)
    tissue_params[:, 2 * N_LAYERS + 2] = random_state.uniform(G_MIN, G_MAX, n_samples)

    # n_2, n_3 (layers 1, 2)
    tissue_params[:, 3 * N_LAYERS + 1 : 3 * N_LAYERS + 3] = random_state.uniform(
        N_MIN, N_MAX, (n_samples, 2)
    )

    # d_2, d_3 (layers 1, 2)
    tissue_params[:, 4 * N_LAYERS + 1 : 4 * N_LAYERS + 3] = (
        random_state.uniform(sqrt_d_min, sqrt_d_max, (n_samples, 2))
    ) ** 2

    # Reshape data into expected shape (batch_size, n_wvls, n_params)
    tissue_params = tissue_params.reshape(-1, n_wvl, n_layers * N_PARAMS)

    # Make bottom-most layer sufficiently thick
    if infinite_thickness:
        tissue_params[:, :, -1] = D_MAX * 100  # equals 20cm

    # Validate output
    expected_shape = (n_samples // n_wvl, n_wvl, N_LAYERS * N_PARAMS)
    if tissue_params.shape != expected_shape:
        raise ValueError(
            f"Parameter array has shape {tissue_params.shape}, "
            f"expected {expected_shape}"
        )
    if (tissue_params == 0).any():
        raise ValueError("Some parameters have not been properly initialized")

    logger.info(f"Generated hand-crafted parameters: {tissue_params.shape}")
    logger.info(
        f"Parameter ranges - mu_a: [{tissue_params[:, :, :N_LAYERS].min():.2e},"
        f" {tissue_params[:, :, :N_LAYERS].max():.2e}]"
    )
    logger.info(
        "Parameter ranges - mu_s: "
        f"[{tissue_params[:, :, N_LAYERS : 2 * N_LAYERS].min():.2e},"
        f" {tissue_params[:, :, N_LAYERS : 2 * N_LAYERS].max():.2e}]"
    )

    return tissue_params


def simulate_and_save_physical_simulations(
    physical_params: np.ndarray,
    batch_size: int,
    save_str: str,
    run_id: int | None = None,
    timeout: float | None = None,
    nr_photons: int = NR_PHOTONS,
) -> None:
    """Run Monte Carlo simulations and save results.

    Args:
        physical_params: Physical parameters for simulation
        batch_size: Number of samples per simulation batch
        save_str: String identifier for saving results
        run_id: Run identifier for file naming
        timeout: Timeout for simulations in seconds
        nr_photons: Number of photons for Monte Carlo simulation

    Raises:
        ValueError: If batch size does not match or number of parameters does not match.
        FileNotFoundError: If output directory cannot be created.
    """
    logger.info(f"Starting simulations for {len(physical_params)} parameter sets")
    dummy_wavelength = 400 * 10**-9

    # Process in batches
    for i in track(
        range(0, len(physical_params), batch_size), description="Simulating batches"
    ):
        batch_params = physical_params[i : i + batch_size]
        batch_filename = f"{save_str}/batch_{run_id}_{i}.txt"
        batch_path = os.path.join(OUTPUT_DIR, batch_filename)

        # Create output directory
        os.makedirs(os.path.dirname(batch_path), exist_ok=True)

        try:
            # Skip if batch already exists
            data = np.loadtxt(batch_path, delimiter=",")
            if len(data) != batch_size:
                raise ValueError(
                    "Batch size does not match! Check given batch size "
                    "and number of samples."
                )
            if data.shape[1] != N_LAYERS * N_PARAMS + N_WVL:
                raise ValueError(
                    "Number of parameters does not match! Check number "
                    "of parameters and number of pseudo wavelengths."
                )
        except FileNotFoundError:
            # Run Monte Carlo simulation
            results = calculate_spectrum_for_physical_batch(
                pd.DataFrame(
                    reorder_parameters(
                        batch_params.reshape(-1, N_LAYERS * N_PARAMS), N_LAYERS
                    ),
                    columns=generate_simulation_header(N_LAYERS, dummy_wavelength),
                ),
                wavelengths=np.array([dummy_wavelength]),
                nr_photons=nr_photons,
                mci_base_folder=os.path.join(
                    OUTPUT_DIR, f"{save_str}/mci_{run_id}_{i}"
                ),
                batch_id=str(i),
                ignore_a=IGNORE_A,
                mco_file=os.path.join(
                    OUTPUT_DIR, f"{save_str}/mci_{run_id}_{i}/batch.mco"
                ),
                verbose=VERBOSE,
                timeout=timeout,
            )

            # Concatenate and save results
            data = np.concatenate(
                [
                    batch_params.reshape(batch_size, -1),
                    results.reflectances.to_numpy().reshape(batch_size, -1),
                ],
                axis=1,
            )
            np.savetxt(batch_path, data, delimiter=",")
            logger.info(f"Saved batch {i} to {batch_path}")


def main() -> None:
    """Main function to orchestrate the data generation process."""
    # Parse and validate arguments
    args = parse_arguments()
    validate_arguments(args)

    logger.info(
        f"Starting physical data generation of {args.run_id} of {args.n_runs} runs."
    )
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # Generate physical parameters
    physical_params = hand_crafted_layer_generation(
        n_samples=args.n_samples,
        n_wvl=N_WVL,
        n_layers=N_LAYERS,
        seed=42,
    )

    # Get consistent batch if run_id is specified
    if args.run_id is not None:
        physical_params = get_consistent_parameter_batch(
            physical_params, run_id=args.run_id, n_runs=args.n_runs
        )

    # Run simulations and save results
    save_str = "hand_crafted_3_layer_100_Mio_thick"
    simulate_and_save_physical_simulations(
        physical_params=physical_params,
        batch_size=args.batch_size,
        save_str=save_str,
        run_id=args.run_id,
        nr_photons=args.nr_photons,
    )


if __name__ == "__main__":
    main()
