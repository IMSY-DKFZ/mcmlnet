from itertools import product

import numpy as np
import pytest
import torch

from mcmlnet.utils.knn_cuml import CuMLKNeighbors


class TestCuMLKNeighbors:
    """Test cases for the CuMLKNeighbors class."""

    def setup_method(self) -> None:
        """Setup method to initialize common variables."""
        self.X_float32 = np.random.rand(10, 3).astype(np.float32)
        self.X_float64 = np.random.rand(10, 3).astype(np.float64)
        self.X_torch = torch.rand(10, 3)
        self.y = np.array([0, 1, 1, 0, 1, 0, 1, 0, 0, 0])

    @pytest.mark.parametrize(  # type: ignore[misc]
        "input, distance_type",
        list(
            product(
                [
                    np.random.rand(10, 3).astype(np.float32),
                    np.random.rand(10, 3).astype(np.float64),
                    torch.rand(10, 3),
                ],
                ["l1", "l2", "euclidean", "cosine", "correlation"],
            )
        ),
    )
    def test_knn_fit_and_kneighbors(
        self, input: np.ndarray | torch.Tensor, distance_type: str
    ) -> None:
        """Test fitting and querying the KNN model."""
        knn = CuMLKNeighbors(k=3, distance_type=distance_type)
        knn.fit(input)
        indices, distances = knn.kneighbors(input)
        assert indices.shape == (10, 3)
        assert distances.shape == (10, 3)

    def test_knn_invalid_distance_type(self) -> None:
        """Test initialization with an invalid distance type."""
        with pytest.raises(ValueError, match="Unsupported distance type:"):
            CuMLKNeighbors(distance_type="not_supported")

    @pytest.mark.parametrize(  # type: ignore[misc]
        "distance_type", ["l1", "l2", "euclidean", "cosine", "correlation"]
    )
    def test_knn_more_neighbors_than_samples(self, distance_type: str) -> None:
        """Test requesting more neighbors than samples."""
        knn = CuMLKNeighbors(k=12, distance_type=distance_type)
        knn.fit(self.X_float32)
        with pytest.raises(
            ValueError, match="n_neighbors must be <= number of samples in index"
        ):
            _, _ = knn.kneighbors(self.X_float32)

    def test_knn_not_fitted(self) -> None:
        """Test calling kneighbors before fitting the model."""
        knn = CuMLKNeighbors(k=2)
        with pytest.raises(
            RuntimeError, match="Model must be fitted before calling kneighbors"
        ):
            knn.kneighbors(np.random.rand(2, 2))

    @pytest.mark.parametrize(  # type: ignore[misc]
        "input",
        [
            np.random.rand(10, 3).astype(np.float32),
            np.random.rand(10, 3).astype(np.float64),
            torch.rand(10, 3),
        ],
    )
    def test_knn_predict(self, input: np.ndarray | torch.Tensor) -> None:
        """Test prediction functionality."""
        knn = CuMLKNeighbors(k=3, distance_type="l2")
        knn.fit(input, self.y)
        preds = knn.predict(input)
        assert preds.shape == (10,)
        assert set(preds).issubset({0, 1})

    def test_knn_predict_without_labels(self) -> None:
        """Test prediction without fitting with labels."""
        knn = CuMLKNeighbors(k=3, distance_type="l2")
        knn.fit(self.X_float32)
        with pytest.raises(
            RuntimeError, match="Labels must be provided during fit to use predict"
        ):
            knn.predict(self.X_float32)

    def test_knn_wrong_dim(self) -> None:
        """Test handling of wrong dimensional input data."""
        knn = CuMLKNeighbors(k=2)
        with pytest.raises(ValueError, match="Data must be 2D, got 1D"):
            knn.fit(self.X_float32[0])
        with pytest.raises(ValueError, match="Data must be 2D, got 3D"):
            knn.fit(self.X_float32[None, :, :])

        knn.fit(self.X_float32)
        with pytest.raises(ValueError, match="Query data must be 2D, got 1D"):
            knn.kneighbors(self.X_float32[0])
        with pytest.raises(ValueError, match="Query data must be 2D, got 3D"):
            knn.kneighbors(self.X_float32[None, :, :])

        knn.fit(self.X_float32, self.y)
        with pytest.raises(ValueError, match="Query data must be 2D, got 1D"):
            knn.predict(self.X_float32[0])
        with pytest.raises(ValueError, match="Query data must be 2D, got 3D"):
            knn.predict(self.X_float32[None, :, :])

    def test_knn_fit_verbose(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test verbose output during fitting."""
        caplog.clear()
        caplog.set_level("INFO")
        knn = CuMLKNeighbors(k=3, distance_type="l2", verbose=False)
        knn.fit(self.X_float32, verbose=True)
        assert "Model fitted in" in caplog.text

    def test_knn_kneighbors_verbose(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test verbose output during kneighbors."""
        caplog.clear()
        caplog.set_level("INFO")
        knn = CuMLKNeighbors(k=3, distance_type="l2", verbose=False)
        knn.fit(self.X_float32, verbose=False)
        knn.kneighbors(self.X_float32, verbose=True)
        assert "Search completed in" in caplog.text

    def test_knn_predict_verbose(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test verbose output during prediction."""
        caplog.clear()
        caplog.set_level("INFO")
        knn = CuMLKNeighbors(k=3, distance_type="l2", verbose=False)
        knn.fit(self.X_float32, self.y, verbose=False)
        knn.predict(self.X_float32, verbose=True)
        assert "Search completed in" in caplog.text
