"""Tests for process_spectra.py"""

import numpy as np
import pytest
import torch

from mcmlnet.utils.process_spectra import coeff_of_variation, r_specular


class TestRSpecular:
    """Test cases for the r_specular function."""

    def setup_method(self) -> None:
        """Setup method to initialize common variables."""
        self.n1 = np.array([1.0, 1.33, 1.0])
        self.n2 = np.array([1.5, 1.4, 1.0])
        self.expected = (self.n1 - self.n2) ** 2 / (self.n1 + self.n2) ** 2

    def test_r_specular_float(self) -> None:
        """Test r_specular with float inputs."""
        for n1, n2, expected in zip(self.n1, self.n2, self.expected, strict=False):
            print(n1, n2)
            print(type(n1), type(n2))
            result = r_specular(n1, n2)
            assert np.isclose(result, expected)

    def test_r_specular_numpy(self) -> None:
        """Test r_specular with numpy inputs."""
        result = r_specular(self.n1, self.n2)
        assert np.allclose(result, self.expected)

    def test_r_specular_torch(self) -> None:
        """Test r_specular with torch inputs."""
        result = r_specular(torch.tensor(self.n1), torch.tensor(self.n2))
        assert torch.allclose(result, torch.tensor(self.expected))


class TestCoeffOfVariation:
    """Test cases for the coeff_of_variation function."""

    def setup_method(self) -> None:
        """Setup method to initialize common variables."""
        self.data_torch = torch.tensor([0.2, 0.5, 0.8])
        self.data_numpy = np.array([0.2, 0.5, 0.8])
        self.n_photons = 1000
        self.expected_torch = torch.sqrt(
            (1 - self.data_torch) / self.data_torch / self.n_photons
        )
        self.expected_numpy = np.sqrt(
            (1 - self.data_numpy) / self.data_numpy / self.n_photons
        )

    def test_coeff_of_variation_valid_torch(self) -> None:
        """Test coeff_of_variation with torch inputs."""
        result = coeff_of_variation(self.data_torch, self.n_photons)
        assert result.shape == self.data_torch.shape
        assert isinstance(result, torch.Tensor)
        assert torch.allclose(result, self.expected_torch)

    def test_coeff_of_variation_valid_numpy(self) -> None:
        """Test coeff_of_variation with numpy inputs."""
        result = coeff_of_variation(self.data_numpy, self.n_photons)
        assert result.shape == self.data_numpy.shape
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, self.expected_numpy)

    @pytest.mark.parametrize(  # type: ignore[misc]
        "data",
        [
            torch.tensor([-0.1, 0.5]),
            torch.tensor([0.5, 1.1]),
            np.array([-0.2, 0.3]),
            np.array([0.7, 1.2]),
        ],
    )
    def test_coeff_of_variation_invalid_data(
        self, data: torch.Tensor | np.ndarray
    ) -> None:
        """Test coeff_of_variation with invalid data inputs."""
        with pytest.raises(ValueError, match="Data must be in the range"):
            coeff_of_variation(data, self.n_photons)

    @pytest.mark.parametrize("n_photons", [0, -1, -100])  # type: ignore[misc]
    def test_coeff_of_variation_invalid_n_photons(self, n_photons: int) -> None:
        """Test coeff_of_variation with invalid n_photons inputs."""
        with pytest.raises(
            ValueError, match="Number of photons must be greater than zero"
        ):
            coeff_of_variation(self.data_torch, n_photons)
