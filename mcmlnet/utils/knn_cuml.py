"""
Fast KNN implementation using cuML.

This implementation leverages NVIDIA's cuML library for GPU-accelerated
nearest neighbor search, providing significant speed-ups for large datasets
while maintaining compatibility with PyTorch tensors.

cuML offers excellent performance on NVIDIA GPUs
and integrates well with the CUDA ecosystem.
"""

import time

import numpy as np
import torch
from cuml.neighbors import NearestNeighbors

from mcmlnet.utils.logging import setup_logging
from mcmlnet.utils.tensor import TensorType

logger = setup_logging(level="info", logger_name=__name__)

# Supported distance types for cuML
SUPPORTED_DISTANCES = {
    "l1": "Manhattan distance (L1)",
    "cityblock": "City block distance",
    "taxicab": "Taxicab distance",
    "manhattan": "Manhattan distance",
    "euclidean": "Euclidean distance",
    "l2": "Euclidean distance (L2)",
    "braycurtis": "Bray-Curtis distance",
    "canberra": "Canberra distance",
    "minkowski": "Minkowski distance with configurable p",
    "chebyshev": "Chebyshev distance (L∞)",
    "jensenshannon": "Jensen-Shannon distance",
    "cosine": "Cosine similarity",
    "correlation": "Correlation distance",
}


class CuMLKNeighbors:
    """
    Fast K-Nearest Neighbors implementation using cuML.

    This class provides efficient nearest neighbor search with GPU acceleration
    using NVIDIA's cuML library, supporting various distance metrics and
    seamless integration with PyTorch tensors.
    """

    def __init__(
        self,
        k: int = 1,
        distance_type: str = "l2",
        algorithm: str = "auto",
        metric_params: dict | None = None,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the cuML KNN model.

        Args:
            k: Number of nearest neighbors to return
            distance_type: Distance metric to use (see SUPPORTED_DISTANCES)
            algorithm: Algorithm to use ('auto', 'rbc', 'brute', 'ivfflat', ...)
            metric_params: Additional parameters for the metric
            verbose: Whether to enable verbose output
        """
        self.k = k
        self.distance_type = distance_type
        self.algorithm = algorithm
        self.metric_params = metric_params or {}
        self.verbose = verbose

        # Validate distance type
        if distance_type not in SUPPORTED_DISTANCES:
            raise ValueError(
                f"Unsupported distance type: {distance_type}. "
                f"Supported types: {list(SUPPORTED_DISTANCES.keys())}"
            )

        # Initialize attributes
        self.knn_model: NearestNeighbors | None = None
        self.y: torch.Tensor | None = None
        self._is_fitted = False

    def _torch_to_numpy(self, data: TensorType) -> np.ndarray:
        """Convert PyTorch tensor to NumPy array."""
        if isinstance(data, torch.Tensor):
            return data.numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def fit(
        self,
        X: TensorType,
        y: TensorType | None = None,
        verbose: bool = True,
    ) -> "CuMLKNeighbors":
        """
        Fit the KNN model with training data.

        Args:
            X: Training data (2D array/tensor)
            y: Optional labels for prediction
            verbose: Whether to show timing information

        Returns:
            Self for method chaining
        """
        if X.ndim != 2:
            raise ValueError(f"Data must be 2D, got {X.ndim}D")

        start_time = time.time()

        # Create and fit cuML NearestNeighbors model
        self.knn_model = NearestNeighbors(
            n_neighbors=self.k,
            metric=self.distance_type,
            algorithm=self.algorithm,
            metric_params=self.metric_params,
            p=self.metric_params.get("p", 2.0),
            verbose=self.verbose,
            output_type="numpy",
        )
        self.knn_model.fit(self._torch_to_numpy(X))

        # Store labels if provided
        if y is not None:
            self.y = self._torch_to_numpy(y)

        self._is_fitted = True

        if verbose:
            elapsed = time.time() - start_time
            logger.info(f"Model fitted in {elapsed:.3f}s")

        return self

    def kneighbors(
        self,
        X: TensorType | None = None,
        two_pass_precision: bool = True,
        verbose: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Find k-nearest neighbors for input data.

        Args:
            X: Query data (2D array/tensor). If None, uses training data
            two_pass_precision: Whether to use two-pass precision
            verbose: Whether to show timing information

        Returns:
            Tuple of (indices, distances)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling kneighbors")

        if X is not None and X.ndim != 2:
            raise ValueError(f"Query data must be 2D, got {X.ndim}D")

        start_time = time.time()

        # Find nearest neighbors
        distances, indices = self.knn_model.kneighbors(  # type: ignore [union-attr]
            self._torch_to_numpy(X),
            n_neighbors=self.k,
            return_distance=True,
            two_pass_precision=two_pass_precision,
        )

        if verbose:
            elapsed = time.time() - start_time
            logger.info(f"Search completed in {elapsed:.3f}s")

        return indices, distances

    def predict(
        self,
        X: TensorType,
        two_pass_precision: bool = False,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Predict labels using majority voting from nearest neighbors.

        Args:
            X: Query data (2D array/tensor)
            two_pass_precision: Whether to use two-pass precision
            verbose: Whether to show timing information

        Returns:
            Predicted labels as NumPy array
        """
        if self.y is None:
            raise RuntimeError("Labels must be provided during fit to use predict")

        # Get nearest neighbors
        _, indices = self.kneighbors(
            X, two_pass_precision=two_pass_precision, verbose=verbose
        )
        indices = indices.astype(int)

        # Get votes from nearest neighbors
        votes = self.y[indices]

        # Majority voting
        if isinstance(votes, np.ndarray):
            predictions = np.array([np.argmax(np.bincount(row)) for row in votes])
            return predictions
        else:
            raise RuntimeError("Unexpected data type for votes")

    def __del__(self) -> None:
        """Cleanup resources."""
        # Clear data to prevent accidental reuse
        for attr in ["y", "knn_model"]:
            if hasattr(self, attr):
                delattr(self, attr)
