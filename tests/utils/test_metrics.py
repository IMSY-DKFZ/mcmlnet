"""Tests for mcmlnet.utils.metrics module."""

import numpy as np
import pytest
import torch

from mcmlnet.utils.metrics import (
    compute_is_in_recall,
    custom_correlation_coefficient_diagonal,
    custom_r2_score,
    mape,
    nmae,
    nrmse,
)


class TestMetrics:
    """Test cases for metrics module."""

    def setup_method(self) -> None:
        """Setup common variables for tests."""
        self.targets = torch.tensor([1.0, 2.0, 3.0])

    def test_mape_valid_inputs(self) -> None:
        """Test MAPE calculation with valid inputs."""
        mean_mape, lower_pi, upper_pi = mape(self.targets, self.targets)

        assert mean_mape == 0.0
        assert lower_pi == 0.0
        assert upper_pi == 0.0

    def test_mape_with_errors(self) -> None:
        """Test MAPE calculation with prediction errors."""
        predictions = torch.tensor([1.1, 2.2, 3.3])

        mean_mape, lower_pi, upper_pi = mape(predictions, self.targets)

        assert mean_mape > 0.0
        assert lower_pi >= 0.0
        assert upper_pi >= lower_pi

    def test_mape_shape_mismatch(self) -> None:
        """Test MAPE calculation with shape mismatch."""
        predictions = torch.tensor([1.0, 2.0])

        with pytest.raises(ValueError, match="Shape mismatch"):
            mape(predictions, self.targets)

    def test_mape_non_finite_values(self) -> None:
        """Test MAPE calculation with non-finite values."""
        predictions = torch.tensor([1.0, np.inf, 3.0])

        with pytest.raises(ValueError, match="non-finite values"):
            mape(predictions, self.targets)

        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = self.targets.clone()
        targets[0] = np.inf

        with pytest.raises(ValueError, match="non-finite values"):
            mape(predictions, targets)

    def test_mape_zero_targets(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test MAPE calculation with zero targets."""
        predictions = torch.tensor([0.1, 0.2, 0.3])
        targets = torch.tensor([0.0, 0.0, 0.0])

        # Clear any existing log records
        caplog.clear()

        # Set log level to capture warnings
        caplog.set_level("WARNING")

        mape(predictions, targets)

        # Check that the warning was logged
        assert "Zero values in targets detected for MAPE calculation" in caplog.text

    def test_nmae_valid_inputs(self) -> None:
        """Test NMAE calculation with valid inputs."""
        mean_nmae, lower_pi, upper_pi = nmae(self.targets, self.targets)

        assert mean_nmae == 0.0
        assert lower_pi == 0.0
        assert upper_pi == 0.0

    def test_nmae_with_errors(self) -> None:
        """Test NMAE calculation with prediction errors."""
        predictions = torch.tensor([1.1, 2.2, 3.3])

        mean_nmae, lower_pi, upper_pi = nmae(predictions, self.targets)

        assert mean_nmae > 0.0
        assert lower_pi >= 0.0
        assert upper_pi >= lower_pi

    def test_nmae_shape_mismatch(self) -> None:
        """Test NMAE calculation with shape mismatch."""
        predictions = torch.tensor([1.0, 2.0])

        with pytest.raises(ValueError, match="Shape mismatch"):
            nmae(predictions, self.targets)

    def test_nmae_non_finite_values(self) -> None:
        """Test NMAE calculation with non-finite values."""
        predictions = torch.tensor([1.0, np.inf, 3.0])

        with pytest.raises(ValueError, match="non-finite values"):
            nmae(predictions, self.targets)

        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = self.targets.clone()
        targets[0] = np.inf

        with pytest.raises(ValueError, match="non-finite values"):
            nmae(predictions, targets)

    def test_nmae_zero_targets(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test NMAE calculation with zero targets."""
        predictions = torch.tensor([0.1, 0.2, 0.3])
        targets = torch.tensor([0.0, 0.0, 0.0])

        # Clear any existing log records
        caplog.clear()

        # Set log level to capture warnings
        caplog.set_level("WARNING")

        nmae(predictions, targets)

        # Check that the warning was logged
        assert "Zero values in targets detected for NMAE calculation" in caplog.text

    def test_nrmse_valid_inputs(self) -> None:
        """Test NRMSE calculation with valid inputs."""
        mean_nrmse, lower_pi, upper_pi = nrmse(self.targets, self.targets)

        assert mean_nrmse == 0.0
        assert lower_pi == 0.0
        assert upper_pi == 0.0

    def test_nrmse_with_errors(self) -> None:
        """Test NRMSE calculation with prediction errors."""
        predictions = torch.tensor([1.1, 2.2, 3.3])

        mean_nrmse, lower_pi, upper_pi = nrmse(predictions, self.targets)

        assert mean_nrmse > 0.0
        assert lower_pi >= 0.0
        assert upper_pi >= lower_pi

    def test_nrmse_shape_mismatch(self) -> None:
        """Test NRMSE calculation with shape mismatch."""
        predictions = torch.tensor([1.0, 2.0])

        with pytest.raises(ValueError, match="Shape mismatch"):
            nrmse(predictions, self.targets)

    def test_nrmse_non_finite_values(self) -> None:
        """Test NRMSE calculation with non-finite values."""
        predictions = torch.tensor([1.0, np.inf, 3.0])

        with pytest.raises(ValueError, match="non-finite values"):
            nrmse(predictions, self.targets)

        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = self.targets.clone()
        targets[0] = np.inf

        with pytest.raises(ValueError, match="non-finite values"):
            nrmse(predictions, targets)

    def test_nrmse_zero_targets(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test NRMSE calculation with zero targets."""
        predictions = torch.tensor([0.1, 0.2, 0.3])
        targets = torch.tensor([0.0, 0.0, 0.0])

        # Clear any existing log records
        caplog.clear()

        # Set log level to capture warnings
        caplog.set_level("WARNING")

        nrmse(predictions, targets)

        # Check that the warning was logged
        assert "Zero values in targets detected for NRMSE calculation" in caplog.text

    def test_r2_score_valid_inputs(self) -> None:
        """Test R-squared score calculation with valid inputs."""
        r2 = custom_r2_score(self.targets, self.targets)

        assert r2.mean() == 1.0

    def test_r2_score_with_errors(self) -> None:
        """Test R-squared score calculation with prediction errors."""
        predictions = torch.tensor([1.1, 2.2, 3.3])

        r2 = custom_r2_score(self.targets, predictions)

        assert r2.mean() < 1.0  # Should be less than perfect

    def test_r2_score_constant_targets(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test R-squared score calculation with constant targets."""
        predictions = torch.randn(100)
        targets = torch.ones(100)  # Constant

        # Clear any existing log records
        caplog.clear()

        # Set log level to capture warnings
        caplog.set_level("WARNING")

        r2 = custom_r2_score(targets, predictions)

        # R-squared should be very small for constant targets
        assert r2.mean() < 1e-6

        # Check that the warning was logged
        assert (
            "Total error is zero for some samples, "
            "setting to 1 for R2 score calculation" in caplog.text
        )

    def test_r2_score_broadcast(self) -> None:
        """Test R-squared score calculation with broadcasted inputs."""
        predictions = torch.randn(100)
        targets = torch.randn(100, 5)

        with pytest.raises(ValueError, match="Shapes mismatch and cannot broadcast"):
            custom_r2_score(targets, predictions)

        predictions = torch.randn(100, 3)
        targets = torch.randn(100, 5, 3)

        r2 = custom_r2_score(targets, predictions)
        assert r2.shape == (100, 5)

        predictions = torch.randn(100, 5, 3)
        targets = torch.randn(100, 3)

        r2 = custom_r2_score(targets, predictions)
        assert r2.shape == (100, 5)

    def test_metrics_with_numpy_inputs(self) -> None:
        """Test metrics work with numpy arrays."""
        # Test that all metrics work with numpy inputs
        mape_result = mape(self.targets.numpy(), self.targets.numpy())
        nmae_result = nmae(self.targets.numpy(), self.targets.numpy())
        nrmse_result = nrmse(self.targets.numpy(), self.targets.numpy())

        # Check that all return tuples of 3 values
        for result in [mape_result, nmae_result, nrmse_result]:
            assert len(result) == 3
            assert all(isinstance(x, float) for x in result)

    def test_metrics_with_batch_dimensions(self) -> None:
        """Test metrics work with batch dimensions."""
        predictions = torch.randn(10, 5, 3)  # Batch of 10, 5 wavelengths, 3 layers
        targets = torch.randn(10, 5, 3)

        # Test that all metrics work with batch dimensions
        mape_result = mape(predictions, targets)
        nmae_result = nmae(predictions, targets)
        nrmse_result = nrmse(predictions, targets)

        # Check that all return tuples of 3 values
        for result in [mape_result, nmae_result, nrmse_result]:
            assert len(result) == 3
            assert all(isinstance(x, float) for x in result)

        r2_result = custom_r2_score(targets, predictions)
        assert r2_result.shape == (predictions.shape[0], predictions.shape[1])

    def test_metrics_edge_cases(self) -> None:
        """Test metrics with edge cases."""
        # Single element
        predictions = torch.tensor([1.0])
        targets = torch.tensor([1.0])

        mape_result = mape(predictions, targets)
        nmae_result = nmae(predictions, targets)
        nrmse_result = nrmse(predictions, targets)
        r2_result = custom_r2_score(targets, predictions)

        assert mape_result[0] == 0.0
        assert nmae_result[0] == 0.0
        assert nrmse_result[0] == 0.0
        assert r2_result.mean() == 1.0

        # Large values
        predictions = torch.tensor([1e6, 2e6, 3e6])
        targets = torch.tensor([1e6, 2e6, 3e6])

        mape_result = mape(predictions, targets)
        nmae_result = nmae(predictions, targets)
        nrmse_result = nrmse(predictions, targets)
        r2_result = custom_r2_score(targets, predictions)

        assert mape_result[0] == 0.0
        assert nmae_result[0] == 0.0
        assert nrmse_result[0] == 0.0
        assert r2_result.mean() == 1.0

        # Small values
        predictions = torch.tensor([1e-6, 2e-6, 3e-6])
        targets = torch.tensor([1e-6, 2e-6, 3e-6])

        mape_result = mape(predictions, targets)
        nmae_result = nmae(predictions, targets)
        nrmse_result = nrmse(predictions, targets)
        r2_result = custom_r2_score(targets, predictions)

        assert mape_result[0] == 0.0
        assert nmae_result[0] == 0.0
        assert nrmse_result[0] == 0.0
        assert r2_result.mean() == 1.0


class TestCustomCorrelationCoefficientDiagonal:
    """Test cases for custom_correlation_coefficient_diagonal function."""

    def setup_method(self) -> None:
        """Setup common variables for tests."""
        self.y_true = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def test_perfect_correlation(self) -> None:
        """Test correlation coefficient with perfectly correlated data."""
        corr = custom_correlation_coefficient_diagonal(self.y_true, self.y_true)

        assert corr.shape == (3,)
        assert isinstance(corr, torch.Tensor)
        assert np.allclose(corr, 1.0)

    def test_negative_correlation(self) -> None:
        """Test correlation coefficient with negatively correlated data."""
        corr = custom_correlation_coefficient_diagonal(self.y_true, self.y_true * -1)

        assert corr.shape == (3,)
        assert isinstance(corr, torch.Tensor)
        assert np.allclose(corr, -1.0)

    def test_no_correlation(self) -> None:
        """Test correlation coefficient with uncorrelated data."""
        y_true = torch.randn(100000, 3)
        y_pred = torch.randn(100000, 3)

        corr = custom_correlation_coefficient_diagonal(y_true, y_pred)

        assert corr.shape == (3,)
        assert isinstance(corr, torch.Tensor)
        assert np.allclose(corr, 0.0, atol=1e-2)

    def test_large_dataset(self) -> None:
        """Test correlation coefficient with larger dataset."""
        y_true = torch.randn(100000, 100)
        y_pred = y_true + 0.1 * torch.randn_like(y_true)

        corr = custom_correlation_coefficient_diagonal(y_true, y_pred)

        assert corr.shape == (100,)
        assert isinstance(corr, torch.Tensor)
        # Should be high correlation but not perfect
        assert torch.all(corr > 0.9)
        assert torch.all(corr < 1.0)

    def test_numpy_inputs(self) -> None:
        """Test correlation coefficient with numpy inputs."""
        corr = custom_correlation_coefficient_diagonal(
            self.y_true.numpy(), self.y_true.numpy()
        )

        assert corr.shape == (3,)
        assert isinstance(corr, np.ndarray)
        assert np.allclose(corr, 1.0)

    def test_mixed_inputs(self) -> None:
        """Test correlation coefficient with mixed torch/numpy inputs."""
        corr = custom_correlation_coefficient_diagonal(self.y_true, self.y_true.numpy())

        assert corr.shape == (3,)
        assert isinstance(corr, torch.Tensor)
        assert np.allclose(corr, 1.0)

        corr = custom_correlation_coefficient_diagonal(self.y_true.numpy(), self.y_true)

        assert corr.shape == (3,)
        assert isinstance(corr, np.ndarray)
        assert np.allclose(corr, 1.0)

    def test_single_sample(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test correlation coefficient with single sample."""
        y_true = torch.tensor([[1.0, 2.0, 3.0]])
        y_pred = torch.tensor([[1.0, 2.0, 3.0]])

        # Clear any existing log records
        caplog.clear()

        # Set log level to capture warnings
        caplog.set_level("WARNING")

        corr = custom_correlation_coefficient_diagonal(y_true, y_pred)

        assert corr.shape == (3,)
        assert isinstance(corr, torch.Tensor)
        # With current std=0 handling, correlation should be 0
        assert torch.all(corr == 0.0)
        assert (
            "Standard deviation is zero for some samples, "
            "setting to 1 for correlation coefficient calculation" in caplog.text
        )

    def test_constant_values(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test correlation coefficient with constant values."""
        y_true = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        y_pred = torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])

        # Clear any existing log records
        caplog.clear()

        # Set log level to capture warnings
        caplog.set_level("WARNING")

        corr = custom_correlation_coefficient_diagonal(y_true, y_pred)

        assert corr.shape == (3,)
        assert isinstance(corr, torch.Tensor)
        # With current std=0 handling, correlation should be 0
        assert torch.all(corr == 0.0)
        assert (
            "Standard deviation is zero for some samples, "
            "setting to 1 for correlation coefficient calculation" in caplog.text
        )


class TestComputeIsInRecall:
    """Test cases for compute_is_in_recall function."""

    def setup_method(self) -> None:
        """Setup common variables for tests."""
        self.sim_data = torch.randn(10, 5)
        self.real_data = torch.randn(5, 5)
        self.dist_sim = torch.randn(10, 10)
        self.ids_sim = torch.randint(0, 10, (10, 10))
        self.ids_real = torch.randint(0, 10, (5, 10))
        self.k = 3
        self.metric = "l2"
        self.min_required_r2 = 0.5

    @pytest.mark.parametrize("metric", ["l1", "l2", "cos_sim"])  # type: ignore[misc]
    def test_recall_metrics(self, metric: str) -> None:
        """Test recall computation with different metrics (l1, l2, cos_sim)."""
        recall_check, sim_distances, real_distances = compute_is_in_recall(
            self.sim_data,
            self.real_data,
            self.dist_sim,
            self.ids_sim,
            self.ids_real,
            self.k,
            metric,
            self.min_required_r2,
        )

        assert recall_check.shape == (5,)
        assert sim_distances.shape == (5,)
        assert real_distances.shape == (5,)

    def test_invalid_metric(self) -> None:
        """Test recall computation with invalid metric."""
        metric = "invalid_metric"

        with pytest.raises(ValueError, match="Metric invalid_metric not supported"):
            compute_is_in_recall(
                self.sim_data,
                self.real_data,
                self.dist_sim,
                self.ids_sim,
                self.ids_real,
                self.k,
                metric,
                self.min_required_r2,
            )

    def test_invalid_metric_case_sensitive(self) -> None:
        """Test recall computation with case-sensitive invalid metric."""
        metric = "L2"

        with pytest.raises(ValueError, match="Metric L2 not supported"):
            compute_is_in_recall(
                self.sim_data,
                self.real_data,
                self.dist_sim,
                self.ids_sim,
                self.ids_real,
                self.k,
                metric,
                self.min_required_r2,
            )

    def test_shape_mismatch_validation(self) -> None:
        """Test validation of input shapes."""
        dist_sim = torch.randn(9, 5)  # Wrong length

        with pytest.raises(
            ValueError,
            match=(
                "Simulation data, distance matrix "
                "and indices must have the same length and ndim"
            ),
        ):
            compute_is_in_recall(
                self.sim_data,
                self.real_data,
                dist_sim,
                self.ids_sim,
                self.ids_real,
                self.k,
                self.metric,
                self.min_required_r2,
            )

    def test_real_data_shape_mismatch(self) -> None:
        """Test validation of real data shape mismatch."""
        ids_real = torch.randint(0, 10, (4, 10))  # Wrong length

        with pytest.raises(
            ValueError,
            match=(
                "Simulation data, distance matrix "
                "and indices must have the same length and ndim!"
            ),
        ):
            compute_is_in_recall(
                self.sim_data,
                self.real_data,
                self.dist_sim,
                self.ids_sim,
                ids_real,
                self.k,
                self.metric,
                self.min_required_r2,
            )

    def test_different_wavelength_counts(self) -> None:
        """Test recall computation with different wavelength counts."""
        real_data = torch.randn(5, 7)  # 7 wavelengths (more than sim_data)

        # Should work - it will use only the first 5 wavelengths from real_data
        recall_check, sim_distances, real_distances = compute_is_in_recall(
            self.sim_data,
            real_data,
            self.dist_sim,
            self.ids_sim,
            self.ids_real,
            self.k,
            self.metric,
            self.min_required_r2,
        )

        assert recall_check.shape == (5,)
        assert sim_distances.shape == (5,)
        assert real_distances.shape == (5,)

    def test_edge_case_single_neighbor(self) -> None:
        """Test recall computation with k=1."""
        k = 1

        recall_check, sim_distances, real_distances = compute_is_in_recall(
            self.sim_data,
            self.real_data,
            self.dist_sim,
            self.ids_sim,
            self.ids_real,
            k,
            self.metric,
            self.min_required_r2,
        )

        assert recall_check.shape == (5,)
        assert sim_distances.shape == (5,)
        assert real_distances.shape == (5,)

    def test_high_r2_threshold(self) -> None:
        """Test recall computation with high R2 threshold."""
        min_required_r2 = 0.9999

        recall_check, sim_distances, real_distances = compute_is_in_recall(
            self.sim_data,
            self.real_data,
            self.dist_sim,
            self.ids_sim,
            self.ids_real,
            self.k,
            self.metric,
            min_required_r2,
        )

        assert recall_check.shape == (5,)
        assert sim_distances.shape == (5,)
        assert real_distances.shape == (5,)
        # With very high R2 threshold, distances might be -inf
        # when no points meet criteria
        # This is expected behavior, so we should check
        # that the recall_check handles this
        assert recall_check.dtype == torch.bool
        assert recall_check.sum() == 0

    def test_1d_shape_errors(self) -> None:
        """Test recall computation with 1D data shapes."""
        sim_data = torch.randn(len(self.sim_data))  # 1D tensor

        with pytest.raises(ValueError, match="Simulation data must be a 2D tensor!"):
            compute_is_in_recall(
                sim_data,
                self.real_data,
                self.dist_sim,
                self.ids_sim,
                self.ids_real,
                self.k,
                self.metric,
                self.min_required_r2,
            )

        real_data = torch.randn(5)

        with pytest.raises(ValueError, match="Real data must be a 2D tensor!"):
            compute_is_in_recall(
                self.sim_data,
                real_data,
                self.dist_sim,
                self.ids_sim,
                self.ids_real,
                self.k,
                self.metric,
                self.min_required_r2,
            )

        dist_sim = torch.randn(len(self.sim_data))  # 1D tensor

        with pytest.raises(ValueError, match="Distance matrix must be a 2D tensor!"):
            compute_is_in_recall(
                self.sim_data,
                self.real_data,
                dist_sim,
                self.ids_sim,
                self.ids_real,
                self.k,
                self.metric,
                self.min_required_r2,
            )

        ids_sim = torch.randint(0, 10, (len(self.sim_data),))  # 1D tensor

        with pytest.raises(ValueError, match="Simulation indices must be a 2D tensor!"):
            compute_is_in_recall(
                self.sim_data,
                self.real_data,
                self.dist_sim,
                ids_sim,
                self.ids_real,
                self.k,
                self.metric,
                self.min_required_r2,
            )

        ids_real = torch.randint(0, 10, (self.real_data.shape[0],))  # 1D tensor

        with pytest.raises(ValueError, match="Real indices must be a 2D tensor!"):
            compute_is_in_recall(
                self.sim_data,
                self.real_data,
                self.dist_sim,
                self.ids_sim,
                ids_real,
                self.k,
                self.metric,
                self.min_required_r2,
            )

    def test_3d_shape_errors(self) -> None:
        """Test recall computation with 3D data shapes."""
        sim_data = torch.randn(10, 5, 3)  # 3D tensor

        with pytest.raises(ValueError, match="Simulation data must be a 2D tensor!"):
            compute_is_in_recall(
                sim_data,
                self.real_data,
                self.dist_sim,
                self.ids_sim,
                self.ids_real,
                self.k,
                self.metric,
                self.min_required_r2,
            )

        real_data = torch.randn(5, 5, 3)  # 3D tensor

        with pytest.raises(ValueError, match="Real data must be a 2D tensor!"):
            compute_is_in_recall(
                self.sim_data,
                real_data,
                self.dist_sim,
                self.ids_sim,
                self.ids_real,
                self.k,
                self.metric,
                self.min_required_r2,
            )

        dist_sim = torch.randn(10, 10, 2)  # 3D tensor

        with pytest.raises(ValueError, match="Distance matrix must be a 2D tensor!"):
            compute_is_in_recall(
                self.sim_data,
                self.real_data,
                dist_sim,
                self.ids_sim,
                self.ids_real,
                self.k,
                self.metric,
                self.min_required_r2,
            )

        ids_sim = torch.randint(0, 10, (10, 10, 2))  # 3D tensor

        with pytest.raises(ValueError, match="Simulation indices must be a 2D tensor!"):
            compute_is_in_recall(
                self.sim_data,
                self.real_data,
                self.dist_sim,
                ids_sim,
                self.ids_real,
                self.k,
                self.metric,
                self.min_required_r2,
            )

        ids_real = torch.randint(0, 10, (5, 10, 2))  # 3D tensor

        with pytest.raises(ValueError, match="Real indices must be a 2D tensor!"):
            compute_is_in_recall(
                self.sim_data,
                self.real_data,
                self.dist_sim,
                self.ids_sim,
                ids_real,
                self.k,
                self.metric,
                self.min_required_r2,
            )

    def test_single_sample_edge_case(self) -> None:
        """Test recall computation with single sample data."""
        sim_data = torch.randn(1, 3)
        real_data = torch.randn(1, 3)
        dist_sim = torch.randn(1, 1)
        ids_sim = torch.randint(0, 1, (1, 1))
        ids_real = torch.randint(0, 1, (1, 1))

        recall_check, sim_distances, real_distances = compute_is_in_recall(
            sim_data,
            real_data,
            dist_sim,
            ids_sim,
            ids_real,
            self.k,
            self.metric,
            self.min_required_r2,
        )

        assert recall_check.shape == (1,)
        assert sim_distances.shape == (1,)
        assert real_distances.shape == (1,)
        assert recall_check.dtype == torch.bool

    def test_single_wavelength_edge_case(self) -> None:
        """Test recall computation with single wavelength data."""
        sim_data = torch.randn(10, 1)
        real_data = torch.randn(5, 1)
        dist_sim = torch.randn(10, 10)
        ids_sim = torch.randint(0, 10, (10, 10))
        ids_real = torch.randint(0, 10, (5, 10))

        recall_check, sim_distances, real_distances = compute_is_in_recall(
            sim_data,
            real_data,
            dist_sim,
            ids_sim,
            ids_real,
            self.k,
            self.metric,
            self.min_required_r2,
        )

        assert recall_check.shape == (5,)
        assert sim_distances.shape == (5,)
        assert real_distances.shape == (5,)

    def test_k_larger_than_data_edge_case(self) -> None:
        """Test recall computation with k larger than available data."""
        k = 8

        recall_check, sim_distances, real_distances = compute_is_in_recall(
            self.sim_data,
            self.real_data,
            self.dist_sim,
            self.ids_sim,
            self.ids_real,
            k,
            self.metric,
            self.min_required_r2,
        )

        assert recall_check.shape == (5,)
        assert sim_distances.shape == (5,)
        assert real_distances.shape == (5,)

    @pytest.mark.parametrize("k", [0, -1])  # type: ignore[misc]
    def test_k_edge_cases(self, k: int) -> None:
        """Test recall computation with k=0 and k=-1."""
        with pytest.raises((ValueError, IndexError)):
            compute_is_in_recall(
                self.sim_data,
                self.real_data,
                self.dist_sim,
                self.ids_sim,
                self.ids_real,
                k,
                self.metric,
                self.min_required_r2,
            )
