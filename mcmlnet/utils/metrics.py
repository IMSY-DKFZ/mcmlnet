"""
Utility functions for computing and reporting fit quality metrics
such as MAPE, NMAE, and NRMSE.
"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from mcmlnet.utils.logging import setup_logging
from mcmlnet.utils.tensor import (
    TensorType,
    array_conversion_decorator,
    tensor_conversion_decorator,
)

logger = setup_logging(level="info", logger_name=__name__)

# Constants
EPSILON = 1e-8
QUANTILE_LOWER = 0.025
QUANTILE_UPPER = 0.975
SUPPORTED_METRICS = {"l1", "l2", "cos_sim", "r2"}
SUPPORTED_RECALL_METRICS = {"l1", "l2", "cos_sim"}


def _validate_inputs(predictions: torch.Tensor, targets: torch.Tensor) -> None:
    """Validate input tensors for metric computation.

    Args:
        predictions: Predicted values
        targets: Target values

    Raises:
        ValueError: If inputs contain invalid values
    """
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs "
            f"targets {targets.shape}"
        )

    if not torch.isfinite(predictions).all():
        raise ValueError("Predictions contain non-finite values")
    if not torch.isfinite(targets).all():
        raise ValueError("Targets contain non-finite values")


def _extract_values(tensor: torch.Tensor) -> tuple[float, float, float]:
    """Extract scalar values from tensor operations.

    Args:
        tensor: Input tensor with computed values

    Returns:
        Tuple of (mean, lower_quantile, upper_quantile)
    """
    mean_val = tensor.mean().item()

    try:
        lower_pi = torch.quantile(tensor, QUANTILE_LOWER).item()
        upper_pi = torch.quantile(tensor, QUANTILE_UPPER).item()
    except RuntimeError as e:
        if "too large" in str(e):
            # Fall back to numpy for large tensors
            tensor_np = tensor.detach().cpu().numpy().flatten()
            lower_pi = float(np.quantile(tensor_np, QUANTILE_LOWER))
            upper_pi = float(np.quantile(tensor_np, QUANTILE_UPPER))
        else:
            raise

    return mean_val, lower_pi, upper_pi


def _check_zero_targets(targets: torch.Tensor, metric_name: str) -> None:
    """Check for zero values in targets and log warning if found.

    Args:
        targets: Target values
        metric_name: Name of the metric for logging
    """
    if (targets == 0).any():
        logger.warning(f"Zero values in targets detected for {metric_name} calculation")


@tensor_conversion_decorator
def mape(predictions: TensorType, targets: TensorType) -> tuple[float, float, float]:
    """
    Compute the Mean Absolute Percentage Error (MAPE) between predictions and targets.

    Args:
        predictions: Predicted values (numpy array or torch tensor)
        targets: Target values (numpy array or torch tensor)

    Returns:
        Tuple of (mean_mape, lower_quantile_2.5%, upper_quantile_97.5%)

    Raises:
        ValueError: If inputs have mismatched shapes or contain invalid values
    """
    _validate_inputs(predictions, targets)
    _check_zero_targets(targets, "MAPE")

    ape = torch.abs(predictions - targets) / torch.abs(targets) * 100
    return _extract_values(ape)


@tensor_conversion_decorator
def nmae(predictions: TensorType, targets: TensorType) -> tuple[float, float, float]:
    """
    Compute the Normalized Mean Absolute Error (NMAE) between predictions and targets.

    Args:
        predictions: Predicted values (numpy array or torch tensor)
        targets: Target values (numpy array or torch tensor)

    Returns:
        Tuple of (mean_nmae, lower_quantile_2.5%, upper_quantile_97.5%)

    Raises:
        ValueError: If inputs have mismatched shapes or contain invalid values
    """
    _validate_inputs(predictions, targets)
    _check_zero_targets(targets, "NMAE")

    ae = torch.abs(predictions - targets)
    nae = ae / torch.abs(targets)
    return _extract_values(nae)


@tensor_conversion_decorator
def nrmse(predictions: TensorType, targets: TensorType) -> tuple[float, float, float]:
    """
    Compute the Normalized Root Mean Squared Error (NRMSE)
    between predictions and targets.

    Args:
        predictions: Predicted values (numpy array or torch tensor)
        targets: Target values (numpy array or torch tensor)

    Returns:
        Tuple of (nrmse, lower_quantile_2.5%, upper_quantile_97.5%)

    Raises:
        ValueError: If inputs have mismatched shapes or contain invalid values
    """
    _validate_inputs(predictions, targets)
    _check_zero_targets(targets, "NRMSE")

    se = (predictions - targets) ** 2
    st = targets**2
    rse = torch.sqrt(se) / torch.sqrt(st)

    nrmse_val = torch.sqrt(torch.sum(se)) / torch.sqrt(torch.sum(st))
    _, lower_val, upper_val = _extract_values(rse)

    return nrmse_val.item(), lower_val, upper_val


@tensor_conversion_decorator
def compute_and_report_fit_quality(
    predictions: TensorType,
    targets: TensorType,
    model_name: str,
    print_results: bool = True,
) -> dict[str, Any]:
    """Compute and report the fit quality metrics for the given predictions and targets.

    Args:
        predictions: Predicted values (numpy array or torch tensor)
        targets: Target values (numpy array or torch tensor)
        model_name: Name of the model for reporting
        print_results: Whether to print results to console

    Returns:
        Dictionary containing all computed metrics

    Raises:
        ValueError: If inputs are invalid
    """
    mape_value, mape_lower, mape_upper = mape(predictions, targets)
    nmae_value, nmae_lower, nmae_upper = nmae(predictions, targets)
    nrmse_value, nrmse_lower, nrmse_upper = nrmse(predictions, targets)

    results = {
        "model_name": model_name,
        "mape": {"mean": mape_value, "lower": mape_lower, "upper": mape_upper},
        "nmae": {"mean": nmae_value, "lower": nmae_lower, "upper": nmae_upper},
        "nrmse": {"mean": nrmse_value, "lower": nrmse_lower, "upper": nrmse_upper},
    }

    if print_results:
        print(f"Fit quality for {model_name}:")
        print(f"  MAPE: {mape_value:.4f}% [{mape_lower:.4f}%, {mape_upper:.4f}%]")
        print(f"  NMAE: {nmae_value:.6f} [{nmae_lower:.6f}, {nmae_upper:.6f}]")
        print(f"  NRMSE: {nrmse_value:.6f} [{nrmse_lower:.6f}, {nrmse_upper:.6f}]")

    return results


def _broadcast_tensors(
    tensor1: torch.Tensor, tensor2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcast tensors to compatible shapes.

    Args:
        tensor1: First tensor
        tensor2: Second tensor

    Returns:
        Tuple of broadcasted tensors

    Raises:
        ValueError: If tensors cannot be broadcasted
    """
    if tensor1.shape == tensor2.shape:
        return tensor1, tensor2

    try:
        # Attempt broadcasting (e.g., y_true is (N, W) and y_pred is (N, K, W))
        if tensor1.ndim == 2 and tensor2.ndim == 3:
            tensor1 = tensor1.unsqueeze(1).expand_as(tensor2)
        elif tensor2.ndim == 2 and tensor1.ndim == 3:
            tensor2 = tensor2.unsqueeze(1).expand_as(tensor1)
        else:
            raise RuntimeError("Cannot broadcast tensors")
    except RuntimeError as e:
        raise ValueError(
            f"Shapes mismatch and cannot broadcast: {tensor1.shape} vs {tensor2.shape}"
        ) from e

    return tensor1, tensor2


def custom_r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Compute the R^2 score, vectorized with custom averaging along the spectral axis.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        R^2 scores
    """
    y_true, y_pred = _broadcast_tensors(y_true, y_pred)

    residual_error = torch.sum((y_true - y_pred) ** 2, dim=-1)
    total_error = torch.sum(
        (y_true - torch.mean(y_true, dim=-1, keepdim=True)) ** 2, dim=-1
    )

    # Avoid division by zero
    if torch.any(total_error == 0):
        logger.warning(
            "Total error is zero for some samples, "
            "setting to 1 for R2 score calculation"
        )
    total_error = torch.where(
        total_error == 0, torch.ones_like(total_error), total_error
    )

    return 1 - residual_error / total_error


@array_conversion_decorator
def custom_correlation_coefficient_diagonal(
    y_true: TensorType, y_pred: TensorType
) -> TensorType:
    """
    Compute the diagonal of the correlation coefficient
    between true and predicted values.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Diagonal of the correlation coefficient
    """
    # Compute the correlation coefficient
    true_mean = np.mean(y_true, axis=0)[None, :]
    pred_mean = np.mean(y_pred, axis=0)[None, :]

    true_centered = y_true - true_mean
    pred_centered = y_pred - pred_mean

    covariance_diagonal = np.mean(true_centered * pred_centered, axis=0)
    true_std = np.std(y_true, axis=0)
    pred_std = np.std(y_pred, axis=0)

    if np.any(true_std == 0) or np.any(pred_std == 0):
        logger.warning(
            "Standard deviation is zero for some samples, "
            "setting to 1 for correlation coefficient calculation"
        )
        true_std = np.where(true_std == 0, 1, true_std)
        pred_std = np.where(pred_std == 0, 1, pred_std)

    return covariance_diagonal / (true_std * pred_std)


def compute_distance_metric(
    reference_data: torch.Tensor,
    target_data: torch.Tensor,
    metric: str,
) -> torch.Tensor:
    """Compute a distance or similarity metric between reference and target data.

    Args:
        reference_data: Reference data tensor
        target_data: Target data tensor
        metric: Metric type ('l1', 'l2', 'cos_sim', 'r2')

    Returns:
        Computed distance/similarity metric

    Raises:
        ValueError: If inputs are invalid or metric not supported
    """
    if reference_data.ndim == 1 or target_data.ndim == 1:
        raise ValueError("Reference and target data must be at least 2D tensors.")

    if reference_data.ndim not in [2, 3]:
        raise ValueError(f"Expected 2D or 3D tensors, got {reference_data.ndim}D")

    # Ensure shapes match for broadcasting
    reference_data, target_data = _broadcast_tensors(reference_data, target_data)

    if metric == "l1":
        return F.l1_loss(reference_data, target_data, reduction="none").mean(dim=-1)
    elif metric == "l2":
        return F.mse_loss(reference_data, target_data, reduction="none").mean(dim=-1)
    elif metric == "cos_sim":
        return F.cosine_similarity(reference_data, target_data, dim=-1)
    elif metric == "r2":
        return custom_r2_score(reference_data, target_data)
    else:
        raise ValueError(
            f"Metric {metric} not supported! Supported metrics: {SUPPORTED_METRICS}"
        )


def _validate_recall_inputs(
    sim_data: torch.Tensor,
    real_data: torch.Tensor,
    dist_sim: torch.Tensor,
    ids_sim: torch.Tensor,
    ids_real: torch.Tensor,
    k: int,
    metric: str,
) -> None:
    """Validate inputs for recall computation.

    Args:
        sim_data: Simulation data tensor
        real_data: Real data tensor
        dist_sim: Precomputed distances for simulation data
        ids_sim: Indices of simulation neighbors
        ids_real: Indices of nearest simulation neighbors for real data
        k: Number of nearest neighbors
        metric: Metric type ('l1', 'l2', 'cos_sim')

    Raises:
        ValueError: If inputs are invalid
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if metric not in SUPPORTED_RECALL_METRICS:
        raise ValueError(
            f"Metric {metric} not supported! "
            f"Supported metrics: {SUPPORTED_RECALL_METRICS}"
        )

    if sim_data.ndim != 2:
        raise ValueError("Simulation data must be a 2D tensor!")

    if real_data.ndim != 2:
        raise ValueError("Real data must be a 2D tensor!")

    if dist_sim.ndim != 2:
        raise ValueError("Distance matrix must be a 2D tensor!")

    if ids_sim.ndim != 2:
        raise ValueError("Simulation indices must be a 2D tensor!")

    if ids_real.ndim != 2:
        raise ValueError("Real indices must be a 2D tensor!")

    if (
        len(sim_data) != len(dist_sim)
        or len(sim_data) != len(ids_sim)
        or len(real_data) != len(ids_real)
        or sim_data.ndim != ids_sim.ndim
        or real_data.ndim != ids_real.ndim
        or sim_data.ndim != dist_sim.ndim
        or dist_sim.ndim != ids_sim.ndim
        or dist_sim.ndim != ids_real.ndim
    ):
        raise ValueError(
            "Simulation data, distance matrix and indices "
            "must have the same length and ndim!"
        )

    if len(real_data) != len(ids_real) or real_data.ndim != ids_real.ndim:
        raise ValueError("Real data and indices must have the same length and ndim!")


def _compute_simulation_distances(
    sim_data: torch.Tensor,
    ids_sim: torch.Tensor,
    dist_sim: torch.Tensor,
    k: int,
    metric: str,
    min_required_r2: float,
) -> torch.Tensor:
    """Compute simulation distances for recall calculation.

    Args:
        sim_data: Simulation data tensor
        ids_sim: Indices of simulation neighbors
        dist_sim: Precomputed distances for simulation data
        k: Number of nearest neighbors
        metric: Metric type
        min_required_r2: Minimum required R2 score

    Returns:
        Computed simulation distances
    """
    r2_sim_above_threshold = (
        compute_distance_metric(sim_data, sim_data[ids_sim[:, :k]], "r2")
        > min_required_r2
    )
    masked = dist_sim[:, :k].clone()
    if metric == "cos_sim":
        masked[~r2_sim_above_threshold] = float("inf")
        dist_sim_k, _ = masked.min(dim=1)
    else:
        masked[~r2_sim_above_threshold] = float("-inf")
        dist_sim_k, _ = masked.max(dim=1)

    return dist_sim_k


def _compute_real_distances(
    real_data: torch.Tensor,
    sim_data: torch.Tensor,
    ids_real: torch.Tensor,
    k: int,
    metric: str,
    min_required_r2: float,
    n_wvl: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute real data distances for recall calculation.

    Args:
        real_data: Real data tensor
        sim_data: Simulation data tensor
        ids_real: Indices of nearest simulation neighbors for real data
        k: Number of nearest neighbors
        metric: Metric type
        min_required_r2: Minimum required R2 score
        n_wvl: Number of wavelengths

    Returns:
        Tuple of (best distances, corrected indices)
    """
    dist_real_k = compute_distance_metric(
        real_data[:, :n_wvl], sim_data[ids_real[:, :k]], metric
    )
    r2_real_k = (
        compute_distance_metric(real_data[:, :n_wvl], sim_data[ids_real[:, :k]], "r2")
        > min_required_r2
    )
    masked = dist_real_k.clone()
    if metric == "cos_sim":
        masked[~r2_real_k] = float("-inf")
        dist_real_best, corrected_ids_real = masked.max(dim=1)
    else:
        masked[~r2_real_k] = float("inf")
        dist_real_best, corrected_ids_real = masked.min(dim=1)

    return dist_real_best, corrected_ids_real


def compute_is_in_recall(
    sim_data: torch.Tensor,
    real_data: torch.Tensor,
    dist_sim: torch.Tensor,
    ids_sim: torch.Tensor,
    ids_real: torch.Tensor,
    k: int,
    metric: str,
    min_required_r2: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute recall metrics based on a specified distance/similarity metric
    and R2 threshold.

    NOTE: This implementation is actually closer to the coverage than the recall metric
    https://github.com/clovaai/generative-evaluation-prdc

    Args:
        sim_data: Simulation data tensor (N_sim, N_wavelengths)
        real_data: Real data tensor (N_real, M_wavelengths)
        dist_sim: Precomputed distances/similarities
            for simulation data neighbors (N_sim, K_max)
        ids_sim: Indices of simulation neighbors (N_sim, K_max)
        ids_real: Indices of nearest simulation neighbors for real data (N_real, K_max)
        k: Number of nearest neighbors to consider
        metric: Metric used for distance/similarity calculation ('l1', 'l2', 'cos_sim')
        min_required_r2: Minimum required R2 score for recall

    Returns:
        Tuple of (recall_check, sim_distances, real_distances)

    Raises:
        ValueError: If inputs are invalid
    """
    _validate_recall_inputs(sim_data, real_data, dist_sim, ids_sim, ids_real, k, metric)
    n_wvl = sim_data.shape[1]

    # Compute simulation distances
    dist_sim_k = _compute_simulation_distances(
        sim_data, ids_sim, dist_sim, k, metric, min_required_r2
    )

    # Compute real data distances
    dist_real_best, corrected_ids_real = _compute_real_distances(
        real_data, sim_data, ids_real, k, metric, min_required_r2, n_wvl
    )

    # Get the index of the simulation point
    # corresponding to the real data's best neighbor
    best_sim_neighbor_idx = torch.gather(
        ids_real, 1, corrected_ids_real.long().unsqueeze(1)
    ).squeeze(1)

    # Check if the real data point is "closer" to its best sim neighbor
    is_closer = (
        dist_real_best > dist_sim_k[best_sim_neighbor_idx]
        if metric == "cos_sim"
        else dist_real_best < dist_sim_k[best_sim_neighbor_idx]
    )

    # Combine closeness check with real data R2 threshold checks
    final_recall_check = (
        is_closer
        & torch.isfinite(dist_sim_k[best_sim_neighbor_idx])
        & torch.isfinite(dist_real_best)
    )

    return final_recall_check, dist_sim_k[best_sim_neighbor_idx], dist_real_best
