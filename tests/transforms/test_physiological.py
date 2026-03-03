"""Tests for mcmlnet.transforms.physiological module."""

from itertools import product

import numpy as np
import pytest
import torch

from mcmlnet.transforms.physiological import PhysiologicalToPhysicalTransformer


class TestPhysiologicalToPhysicalTransformer:
    """Test cases for PhysiologicalToPhysicalTransformer class."""

    @pytest.mark.parametrize("n_wavelengths", [1, 10, 100, 200, 351])  # type: ignore[misc]
    def test_init_valid_parameters_n_wavelengths(self, n_wavelengths: int) -> None:
        """Test initialization with valid parameters."""
        transformer = PhysiologicalToPhysicalTransformer(
            n_wavelengths=n_wavelengths,
            n_layers=3,
            cHb=150.0,
        )

        assert transformer.n_wavelengths == n_wavelengths
        assert transformer.n_layers == 3
        assert transformer.cHb == 150.0
        assert transformer.eps == 1e-8
        assert transformer.wavelengths.shape == (1, n_wavelengths, 3)
        assert transformer.wavelengths.min() >= 300  # Should start at 300nm or higher
        assert transformer.wavelengths.max() <= 1000  # Should end at 1000nm or lower
        assert transformer.eHbO2.shape == (1, n_wavelengths, 3)
        assert transformer.eHb.shape == (1, n_wavelengths, 3)
        assert pytest.approx(transformer.prefactor) == 2.302585 * 150.0 / 64500

    @pytest.mark.parametrize(  # type: ignore[misc]
        "wavelengths",
        [
            np.array([400, 500, 600, 700, 800]),
            np.linspace(300, 1000, 100),
        ],
    )
    def test_init_valid_parameters_wavelengths(self, wavelengths: np.ndarray) -> None:
        """Test initialization with valid wavelengths."""
        transformer = PhysiologicalToPhysicalTransformer(
            n_wavelengths=len(wavelengths),
            n_layers=3,
            cHb=150.0,
            wavelengths=wavelengths,
        )

        assert transformer.n_wavelengths == len(wavelengths)
        assert transformer.n_layers == 3
        assert transformer.cHb == 150.0
        assert np.array_equal(transformer.wavelengths[0, :, 0], wavelengths)

    def test_init_invalid_n_layers(self) -> None:
        """Test initialization with invalid number of layers."""
        with pytest.raises(ValueError, match="Number of layers must be >= 1"):
            PhysiologicalToPhysicalTransformer(n_wavelengths=100, n_layers=0)

    def test_init_wavelengths_mismatch(self) -> None:
        """Test initialization with wavelength length mismatch."""
        with pytest.raises(
            ValueError, match="Wavelengths length 50 != n_wavelengths 100"
        ):
            PhysiologicalToPhysicalTransformer(
                n_wavelengths=100, wavelengths=np.linspace(400, 1000, 50)
            )

    def test_init_wavelengths_out_of_bounds(self) -> None:
        """Test initialization with wavelengths outside valid range."""
        with pytest.raises(ValueError, match="Some wavelengths outside valid range"):
            PhysiologicalToPhysicalTransformer(
                n_wavelengths=100, wavelengths=np.linspace(200, 1000, 100)
            )
        with pytest.raises(ValueError, match="Some wavelengths outside valid range"):
            PhysiologicalToPhysicalTransformer(
                n_wavelengths=100, wavelengths=np.linspace(300, 5000, 100)
            )

    def test_min_max_logging(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test min_max logging functionality."""
        caplog.clear()
        caplog.set_level("INFO")
        transformer = PhysiologicalToPhysicalTransformer(n_wavelengths=10)
        transformer._min_max(torch.tensor([1.0, 2.0, 3.0]), "test_tensor")

        assert "test_tensor" in caplog.text

    @pytest.mark.parametrize(  # type: ignore[misc]
        "wavelengths, n_layers",
        list(
            product(
                [
                    np.array([400, 500, 600, 700, 800]),
                    np.linspace(300, 1000, 100),
                    np.linspace(300, 1000, 351),
                ],
                [1, 3, 5],
            )
        ),
    )
    def test_transform_physiological_to_physical(
        self, wavelengths: np.ndarray, n_layers: int
    ) -> None:
        """Test transformation from physiological to physical parameters."""
        transformer = PhysiologicalToPhysicalTransformer(
            n_wavelengths=len(wavelengths),
            n_layers=n_layers,
            wavelengths=wavelengths,
        )

        for batch_size in [1, 5, 10]:
            # Create batch of physiological parameters
            physiological_params = (
                torch.tensor(
                    [
                        [0.02, 0.01, 4000, 3, 0.0, 0.9, 1.3, 2] * n_layers,
                    ]
                )
                .repeat(batch_size, 1)
                .squeeze()
            )
            physical_params = transformer.transform_hb_format(physiological_params)

            # Check output shape: (n_samples, n_wavelengths, n_layers * 5)
            expected_shape = (batch_size, len(wavelengths), n_layers * 5)
            assert physical_params.shape == expected_shape
            assert torch.isfinite(physical_params).all()
            assert (physical_params >= 0).all()

    def test_transform_invalid_input_shape(self) -> None:
        """Test transformation with invalid input shape."""
        transformer = PhysiologicalToPhysicalTransformer(n_wavelengths=100, n_layers=3)

        # Invalid shape: 3D input
        invalid_params = torch.rand(5, 10, 24)

        with pytest.raises(ValueError, match="Expected 1 or 2 dimensions, got"):
            transformer.transform_hb_format(invalid_params)

        # Invalid shape: should be (n_samples, n_layers * 8)
        invalid_params = torch.rand(5, 20)

        with pytest.raises(ValueError, match="Expected 24 features, got 20"):
            transformer.transform_hb_format(invalid_params)
