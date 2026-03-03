import os

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
from rich.progress import track
from scipy.optimize import curve_fit

from mcmlnet.training.data_loading.preprocessing import PreProcessor
from mcmlnet.utils.convenience import (
    prepare_surrogate_model_data,
)
from mcmlnet.utils.knn_cuml import CuMLKNeighbors
from mcmlnet.utils.logging import setup_logging
from mcmlnet.utils.metrics import compute_distance_metric

load_dotenv()
logger = setup_logging(level="info", logger_name=__name__)


# region Data preparation / sorting
def compute_reordering_array(
    data: torch.Tensor, raw_data: torch.Tensor, cache_path: str
) -> np.ndarray:
    """Compute and cache the reordering array for 10M data."""
    batch_size = 100
    matched_indices = []
    remaining_indices = list(range(len(raw_data)))

    for i in track(range(0, len(data), batch_size)):
        batch = data[i : i + batch_size, :24].cuda()
        remaining_raw_data = raw_data[remaining_indices, :24].cuda()

        distances = torch.linalg.norm(
            remaining_raw_data[:, None, :] - batch[None, :, :], axis=2
        )
        min_distances, closest_indices = torch.min(distances, dim=0)

        if not torch.all(min_distances == 0):
            logger.warning("Not all closest distances are zero!")

        matched_indices.append([remaining_indices[idx] for idx in closest_indices])
        matched_indices_set = set(matched_indices[-1])
        remaining_indices = list(set(remaining_indices) - matched_indices_set)

    matched_indices = np.concatenate(matched_indices).squeeze()

    np.savetxt(cache_path, matched_indices, fmt="%d")
    return matched_indices


# region Scaling fit
def polynomial_func(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Define the polynomial fit function a * x^b + c."""
    return a * x**b + c


def compute_bootstrap_ci(
    data: np.ndarray,
    confidence_level: float = 0.95,
    n_resamples: int = 1000,
    batch: int | None = 50,
) -> tuple[float, float]:
    """Compute bootstrap confidence intervals for a given dataset."""
    bootstrap_results = stats.bootstrap(
        (data,),
        np.mean,
        n_resamples=n_resamples,
        batch=batch,
        confidence_level=confidence_level,
        method="percentile",
    )
    return (
        bootstrap_results.confidence_interval.low,
        bootstrap_results.confidence_interval.high,
    )


def compute_ci(data: np.ndarray, confidence_level: float = 0.95) -> tuple[float, float]:
    """Compute confidence intervals for a given dataset."""
    return stats.t.interval(  # type: ignore [no-any-return]
        confidence_level, len(data) - 1, loc=np.mean(data), scale=stats.sem(data)
    )


def compute_mae_ci(
    reference_data: np.ndarray, data: np.ndarray, use_logger: bool = True
) -> tuple[float, float]:
    """
    Compute Mean Absolute Error (MAE) confidence intervals.

    Args:
        reference_data: Reference dataset
        data: Comparison dataset
        use_logger: Whether to use logger.info or print

    Returns:
        Tuple of (ci_lower, ci_upper) for 95% confidence interval
    """
    # Ensure both datasets have the same shape
    if reference_data.shape != data.shape:
        raise ValueError("Reference data and comparison data must have the same shape.")

    # Compute absolute error
    ae = np.abs(reference_data - data)

    # Calculate statistics
    quant_25 = np.quantile(ae, 0.025)
    quant_975 = np.quantile(ae, 0.975)

    # Compute 95% CI for the mean
    ci_lower, ci_upper = compute_bootstrap_ci(ae.flatten(), 0.95)

    if use_logger:
        logger.info(f"95% PI: [{quant_25:.4e}, {quant_975:.4e}]")
        logger.info(f"95% CI for Mean: [{ci_lower:.4e}, {ci_upper:.4e}]")
    else:
        print(f"95% PI: [{quant_25:.4e}, {quant_975:.4e}]")
        print(f"95% CI for Mean: [{ci_lower:.4e}, {ci_upper:.4e}]")

    return ci_lower, ci_upper


def neural_scaling_fit(
    x_data: np.ndarray, y_data: np.ndarray, use_logger: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit polynomial function to neural scaling data.

    Args:
        x_data: X values (dataset sizes)
        y_data: Y values (losses)
        use_logger: Whether to use logger.info or print

    Returns:
        Tuple of (x_fit, y_fit, y_fit_min, y_fit_max, popt, perr)
    """
    # Define meaningful bounds
    bounds = ([0, -5, 0], [np.inf, 5, 1e-2])

    # Fit the polynomial function
    popt, pcov = curve_fit(polynomial_func, x_data, y_data, bounds=bounds)

    # Return the fitted data
    x_fit = np.linspace(min(x_data), max(x_data), 100)
    y_fit = polynomial_func(x_fit, *popt)

    # Print statistics
    pcov_str = "pcov:\n" + "\n".join(
        [f"  [{', '.join([f'{val:.7f}' for val in row])}]" for row in pcov]
    )
    if use_logger:
        logger.info("Params for a*x^b + c")
        logger.info(f"popt: {[f'{param:.7f}' for param in popt]}")
        logger.info(pcov_str)
    else:
        print("Params for a*x^b + c")
        print(f"popt: {[f'{param:.7f}' for param in popt]}")
        print(pcov_str)

    perr = np.sqrt(np.diag(pcov))

    if use_logger:
        logger.info(f"perr: {[f'{err:.7f}' for err in perr]}")
    else:
        print(f"perr: {[f'{err:.7f}' for err in perr]}")

    # Return uncertainty bands
    y_fit_min = polynomial_func(x_fit, *(popt - perr))
    y_fit_max = polynomial_func(x_fit, *(popt + perr))

    return x_fit, y_fit, y_fit_min, y_fit_max, popt, perr


def neural_scaling_fit_log_scale(
    x_data: np.ndarray, y_data: np.ndarray, use_logger: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit polynomial function to neural scaling using logarithmic rescaling.

    Args:
        x_data: X values (dataset sizes)
        y_data: Y values (losses)
        use_logger: Whether to use logger.info or print

    Returns:
        Tuple of (x_fit, y_fit, y_fit_min, y_fit_max, popt, perr)
    """

    def log_polynomial_func(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """Define the log polynomial fit function a * x^b + c."""
        return np.log10(polynomial_func(10**x, a, b, c))

    # Define meaningful bounds
    bounds = ([0, -5, 0], [np.inf, 5, 1e-2])

    # Fit the log-polynomial function
    popt, pcov = curve_fit(
        log_polynomial_func, np.log10(x_data), np.log10(y_data), bounds=bounds
    )

    # Return the fitted data
    x_fit = np.linspace(min(x_data), max(x_data), 100)
    y_fit = polynomial_func(x_fit, *popt)

    # Print statistics
    pcov_str = "pcov:\n" + "\n".join(
        [f"  [{', '.join([f'{val:.7f}' for val in row])}]" for row in pcov]
    )
    if use_logger:
        logger.info("Params for a*x^b + c")
        logger.info(f"popt: {[f'{param:.7f}' for param in popt]}")
        logger.info(pcov_str)
    else:
        print("Params for a*x^b + c")
        print(f"popt: {[f'{param:.7f}' for param in popt]}")
        print(pcov_str)

    perr = np.sqrt(np.diag(pcov))

    if use_logger:
        logger.info(f"perr: {[f'{err:.7f}' for err in perr]}")
    else:
        print(f"perr: {[f'{err:.7f}' for err in perr]}")

    # Return uncertainty bands
    y_fit_min = polynomial_func(x_fit, *(popt - perr))
    y_fit_max = polynomial_func(x_fit, *(popt + perr))

    return x_fit, y_fit, y_fit_min, y_fit_max, popt, perr


# region kNN
def fit_knn(
    sim_data: torch.Tensor,
    k: int,
    distance_type: str = "l1",
    verbose: bool = False,
) -> CuMLKNeighbors:
    """
    Fit a CUMl kNN model to simulation data.

    Args:
        sim_data: Simulation data tensor of shape (N_sim, N_features)
        k: Number of nearest neighbors to return
        distance_type: Distance metric to use ('l1', 'l2', 'cosine', 'correlation', ...)
        verbose: Whether to show timing information

    Returns:
        Fitted CuMLKNeighbors model
    """
    knn = CuMLKNeighbors(k=k, distance_type=distance_type, verbose=verbose)
    knn.fit(sim_data, verbose=verbose)
    return knn


def predict_knn(
    knn: CuMLKNeighbors, data: torch.Tensor, verbose: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Find nearest neighbors using a fitted KNN model.

    Args:
        knn: Fitted CuMLKNeighbors model
        data: Query data tensor of shape (N_query, N_features)
        verbose: Whether to show timing information

    Returns:
        Tuple of (indices, distances) where:
        - indices: Indices of nearest neighbors (N_query, k)
        - distances: Distances to nearest neighbors (N_query, k)
    """
    return knn.kneighbors(data, verbose=verbose)


def compute_sim_neighbor_dists(
    sim_data: torch.Tensor,
    knn: CuMLKNeighbors,
    metric: str = "l1",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute distances between simulation points and their k nearest neighbors.

    This function finds the k nearest neighbors for each simulation point and
    computes the specified distance metric between each point and its neighbors.

    Args:
        sim_data: Simulation data tensor of shape (N_sim, N_features)
        knn: Fitted CuMLKNeighbors model
        metric: Distance metric to use ('l1', 'l2', 'cos_sim', 'r2')

    Returns:
        Tuple of (ids_sim, dist_sim) where:
        - ids_sim: Indices of k nearest neighbors for each simulation point (N_sim, k)
        - dist_sim: Distances between each simulation point and its neighbors (N_sim, k)
    """
    # Find k nearest neighbors for each simulation point
    ids_sim, _ = predict_knn(knn, sim_data, verbose=False)

    # (Re-)Compute distances using the specified metric
    dist_sim = compute_distance_metric(sim_data, sim_data[ids_sim], metric)

    return ids_sim, dist_sim


def compute_reflectance_via_nearest_neighbor_lut(
    sim_params: np.ndarray,
    sim_reflectance: np.ndarray,
    real_params: np.ndarray,
    metric: str = "l1",
) -> np.ndarray:
    """
    Compute reflectance values using nearest neighbor lookup table (LUT).

    This function implements a lookup table approach where reflectance values
    for real parameters are estimated by finding the nearest neighbor in the
    simulation parameter space and using its corresponding reflectance value.

    Note: Ensure that input parameters are properly normalized (e.g., using log10
    for mu_a and mu_s' parameters) before calling this function.

    Args:
        sim_params: Simulation parameters array of shape (N_sim, N_params)
        sim_reflectance: Simulation reflectance array of shape (N_sim, N_wavelengths)
        real_params: Real parameters array of shape (N_real, ..., N_params)
        metric: Distance metric to use ('l1', 'l2', 'cos_sim', 'r2')

    Returns:
        Reflectance values corresponding to real parameters, with shape matching
        real_params but with the last dimension replaced by N_wavelengths

    Example:
        >>> sim_params = np.random.randn(1000, 3)  # 1000 simulations, 3 parameters
        >>> sim_reflectance = np.random.randn(
        ...     1000, 100
        ... )  # 1000 simulations, 100 wavelengths
        >>> real_params = np.random.randn(50, 3)  # 50 real samples, 3 parameters
        >>> reflectance = compute_reflectance_via_nearest_neighbor_lut(
        ...     sim_params, sim_reflectance, real_params, metric="l1"
        ... )
        >>> print(reflectance.shape)  # (50, 100)
    """
    # Convert numpy arrays to torch tensors
    sim_params_tensor = torch.from_numpy(sim_params)
    real_params_tensor = torch.from_numpy(real_params)

    # Fit KNN model to simulation parameters
    knn = fit_knn(sim_params_tensor, k=1, distance_type=metric, verbose=False)

    # Find nearest neighbors for real parameters
    ids_real, _ = predict_knn(knn, real_params_tensor, verbose=False)

    # Extract reflectance values for nearest neighbors
    return sim_reflectance[ids_real.squeeze().numpy()]


# region Scaling model evaluation
def validate_model_metadata_file() -> None:
    """
    Validate that the model metadata file exists and has required columns.

    Raises:
        ValueError: If CACHE_DIR environment variable is not set
        FileNotFoundError: If model metadata file doesn't exist
        ValueError: If required columns are missing
    """
    cache_dir = os.getenv("cache_dir")
    if not cache_dir:
        raise ValueError("CACHE_DIR environment variable not set")

    metadata_file = os.path.join(cache_dir, "discovered_scaling_models_df.csv")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(
            f"Model metadata file not found: {metadata_file}! "
            "Please run discover_models.py first."
        )

    # Validate that the CSV has required columns
    try:
        df = pd.read_csv(metadata_file)
        required_columns = ["model_name", "model_path", "model_checkpoint_path", "tdr"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {metadata_file}: {missing_columns}"
            )
    except Exception as err:
        raise ValueError(f"Error reading model metadata file: {err}") from err


def load_model_configs() -> list[dict]:
    """Load model configurations from CSV file."""
    model_path_df = pd.read_csv(
        os.path.join(os.environ["cache_dir"], "discovered_scaling_models_df.csv")
    )

    configs = []
    for _, row in model_path_df.iterrows():
        configs.append(
            {
                "name": row["model_name"],
                "path": row["model_path"],
                "checkpoint_path": row["model_checkpoint_path"],
                "tdr": row["tdr"],
            }
        )

    return configs


def preprocess_datasets_for_model(
    raw_datasets: dict, preprocessor: PreProcessor, cfg: DictConfig
) -> dict:
    """Apply model-specific preprocessing to all datasets."""
    processed_datasets = {}

    for key, raw_data in raw_datasets.items():
        processed_datasets[key] = prepare_surrogate_model_data(
            preprocessor, cfg, raw_data.clone()
        )

    return processed_datasets


def save_results_to_csv(
    results_dict: dict[str, dict[str, float]], filename: str
) -> None:
    """Save results to CSV file."""
    results_df = pd.DataFrame.from_dict(results_dict, orient="index")
    results_df.index.name = "model_name"
    results_df.reset_index(inplace=True)
    print("\nFinal Results")
    print(results_df.head())
    file_path = os.path.join(os.environ["cache_dir"], filename)
    results_df.to_csv(file_path, index=False)
    print(f"Results saved to {file_path}")
