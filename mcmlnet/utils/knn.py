"""
Fast KNN implementation using FAISS.

This implementation is significantly faster than scikit-learn's KNN, achieving
speed-ups of over 1000x for typical use cases (1702s with scikit vs 1.5s on GPU).

Supports multiple distance metrics: L2, L1, cosine similarity, spectral angle
mapping (SAM), weighted L2, and hierarchical navigable small world (HNSW).

NOTE: Reliant on numpy 1.26.4 in version 1.11.0 - thus replaced with cuml.

Useful Links:
- FAISS metrics: https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
- FAISS indexes: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
- Index factory: https://github.com/facebookresearch/faiss/wiki/The-index-factory
"""

import time

import faiss
import faiss.contrib.torch_utils  # adds pytorch tensor support
import numpy as np
import torch

from mcmlnet.utils.logging import setup_logging
from mcmlnet.utils.tensor import TensorType

logger = setup_logging(level="info", logger_name=__name__)

# Supported distance types
SUPPORTED_DISTANCES = {
    "l2": "Euclidean distance",
    "l1": "Manhattan distance",
    "cos_sim": "Cosine similarity",
    "sam": "Spectral angle mapping",
    "weighted_l2": "Weighted Euclidean distance",
    "hsnw": "Hierarchical navigable small world",
}


def l2_normalize(data: TensorType) -> TensorType:
    """L2 normalize data for cosine similarity and SAM calculations."""
    return data / torch.norm(data, dim=-1, keepdim=True)


class FaissKNeighbors:
    """
    Fast K-Nearest Neighbors implementation using FAISS.

    This class provides efficient nearest neighbor search with support for
    multiple distance metrics, GPU acceleration, and weighted distances.
    """

    def __init__(
        self,
        k: int = 1,
        gpu: bool = True,
        distance_type: str = "l2",
        weight: TensorType | None = None,
    ) -> None:
        """
        Initialize the FAISS KNN model.

        Args:
            k: Number of nearest neighbors to return
            gpu: Whether to use GPU acceleration (if available)
            distance_type: Distance metric to use (see SUPPORTED_DISTANCES)
            weight: Weight vector for weighted distance calculations
        """
        self.k = k
        self.distance_type = distance_type
        self.weight = weight
        self.gpu = self._setup_gpu(gpu)

        # Initialize attributes
        self.index = None
        self.data = None
        self.y = None
        self.gpu_resource = None

        # Validate and setup weights if needed
        if "weighted" in distance_type:
            self._validate_weights()

    def _setup_gpu(self, gpu: bool) -> bool:
        """Setup GPU resources if requested and available."""
        if not gpu:
            return False

        if torch.cuda.is_available():
            self.gpu_resource = faiss.StandardGpuResources()
            logger.info("GPU acceleration enabled")
            return True
        else:
            logger.warning("GPU requested but not available. Continuing in CPU mode.")
            return False

    def _validate_weights(self) -> None:
        """Validate weight parameters for weighted distance calculations."""
        if self.weight is None:
            raise ValueError(f"Weights required for {self.distance_type} distance")

        if self.weight.ndim != 1:
            raise ValueError(f"Weights must be 1D array, got {self.weight.ndim}D")

        if not all(self.weight >= 0):
            raise ValueError("Weights must be non-negative")

        if any(self.weight > 10):
            logger.warning("Large weight values detected. Verify this is intended.")

        self.weight = self._to_contiguous(self.weight)

    @staticmethod
    def _to_contiguous(data: TensorType) -> TensorType:
        """
        Convert data to contiguous float32 format for FAISS compatibility.

        Args:
            data: Input tensor or array

        Returns:
            Contiguous float32 data
        """
        if isinstance(data, torch.Tensor):
            return data.float().contiguous()
        elif isinstance(data, np.ndarray):
            return np.ascontiguousarray(data, dtype=np.float32)
        else:
            raise TypeError(
                f"Unsupported data type: {type(data)}. Use torch.Tensor or np.ndarray"
            )

    def _apply_weights(self, data: TensorType, p: float = 2.0) -> TensorType:
        """
        Apply weights to data for weighted distance calculations.

        Args:
            data: Input data
            p: Power for the distance metric

        Returns:
            Weighted data
        """
        if not isinstance(self.weight, type(data)):
            raise ValueError("Data and weights must have the same type")

        if len(self.weight) != data.shape[-1]:
            raise ValueError("Weight dimension must match data feature dimension")

        # Calculate weight power
        if isinstance(self.weight, torch.Tensor):
            weights = torch.float_power(self.weight, exponent=1 / p)
        else:
            weights = np.float_power(self.weight, 1 / p)

        # Broadcast and multiply
        return self._to_contiguous(weights[None, :]) * data

    def _create_index(self, dim: int) -> faiss.Index:
        """Create FAISS index based on distance type."""
        if self.distance_type == "l2":
            return faiss.IndexFlat(dim, faiss.METRIC_L2)
        elif self.distance_type == "l1":
            return faiss.IndexFlat(dim, faiss.METRIC_L1)
        elif self.distance_type in ["cos_sim", "sam"]:
            return faiss.IndexFlat(dim, faiss.METRIC_INNER_PRODUCT)
        elif self.distance_type == "weighted_l2":
            return faiss.IndexFlat(dim, faiss.METRIC_L2)
        elif self.distance_type == "hsnw":
            if self.gpu:
                logger.warning(
                    "HNSW is very slow on GPU (200,000x slower for large datasets)"
                )
            return faiss.IndexHNSWFlat(dim, 32)
        else:
            raise ValueError(
                f"Unsupported distance type: {self.distance_type}. "
                f"Supported types: {list(SUPPORTED_DISTANCES.keys())}"
            )

    def _preprocess_data(self, data: TensorType) -> TensorType:
        """Preprocess data based on distance type."""
        if self.distance_type in ["cos_sim", "sam"]:
            logger.info(f"Applying L2 normalization for {self.distance_type}")
            return l2_normalize(data)
        elif self.distance_type == "weighted_l2":
            return self._apply_weights(data, p=2.0)
        else:
            return data

    def fit(
        self,
        X: TensorType,
        y: TensorType | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Fit the KNN model with training data.

        Args:
            X: Training data (2D array/tensor)
            y: Optional labels for prediction
            verbose: Whether to show timing information
        """
        if X.ndim != 2:
            raise ValueError(f"Data must be 2D, got {X.ndim}D")

        start_time = time.time()

        # Preprocess and store data
        self.data = self._to_contiguous(X)
        self.data = self._preprocess_data(self.data)

        # Create and configure index
        self.index = self._create_index(self.data.shape[-1])  # type: ignore [attr-defined]

        # Move to GPU if requested
        if self.gpu:
            self.index = faiss.index_cpu_to_gpu(self.gpu_resource, 0, self.index)

        # Add data to index
        self.index.add(self.data)  # type: ignore [attr-defined]

        # Store labels if provided
        if y is not None:
            if not isinstance(y, type(self.data)):
                raise ValueError("Labels and data must have the same type")
            if y.ndim != 1:  # type: ignore [attr-defined]
                raise ValueError(f"Labels must be 1D, got {y.ndim}D")  # type: ignore [attr-defined]
            self.y = y

        if verbose:
            elapsed = time.time() - start_time
            logger.info(f"Model fitted in {elapsed:.3f}s")

    def predict(
        self, X: TensorType, verbose: bool = True
    ) -> TensorType | tuple[TensorType, TensorType]:
        """
        Find k-nearest neighbors for input data.

        Args:
            X: Query data (2D array/tensor)
            verbose: Whether to show timing information

        Returns:
            If labels were provided during fit: predicted labels
            Otherwise: (indices, distances) tuple
        """
        if X.ndim != 2:
            raise ValueError(f"Data must be 2D, got {X.ndim}D")

        start_time = time.time()

        # Preprocess query data
        X_processed = self._to_contiguous(X)
        X_processed = self._preprocess_data(X_processed)

        # Search for nearest neighbors
        distances, indices = self.index.search(X_processed, k=self.k)  # type: ignore [attr-defined]

        # Post-process distances for SAM
        if self.distance_type == "sam":
            if isinstance(distances, np.ndarray):
                distances = np.arccos(distances)
            elif isinstance(distances, torch.Tensor):
                distances = torch.acos(distances)

        if verbose:
            elapsed = time.time() - start_time
            logger.info(f"Search completed in {elapsed:.3f}s")

        # Return predictions or indices/distances
        if self.y is not None:
            logger.warning("Label prediction functionality is experimental")  # type: ignore [unreachable]
            return self._predict_labels(indices)
        else:
            return indices, distances

    def _predict_labels(self, indices: TensorType) -> TensorType:
        """Predict labels using majority voting from nearest neighbors."""
        votes = self.y[indices]  # type: ignore [index]

        if isinstance(votes, torch.Tensor):
            votes = votes.numpy()

        if votes.shape != (len(indices), self.k):
            raise ValueError(f"Unexpected votes shape: {votes.shape}")

        # Majority voting
        return np.array([np.argmax(np.bincount(row)) for row in votes])

    def __del__(self) -> None:
        """Cleanup resources and free GPU memory."""
        if hasattr(self, "gpu_resource") and self.gpu_resource:
            del self.gpu_resource  # type: ignore [unreachable]

        # Clear data to prevent accidental reuse
        for attr in ["data", "weight", "index", "y"]:
            if hasattr(self, attr):
                delattr(self, attr)
