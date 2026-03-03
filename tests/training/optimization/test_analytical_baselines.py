"""Tests for mcmlnet.training.optimization.analytical_baselines module."""

import copy

import numpy as np
import pytest
import torch

from mcmlnet.training.optimization.analytical_baselines import (
    AnalyticalModelConstants,
    compute_mu_a,
    compute_mu_s_prime,
    jacques_1999,
    modified_beer_lambert,
)


class TestAnalyticalFunctions:
    """Test cases for analytical baseline functions."""

    def setup_method(self) -> None:
        self.sao2 = torch.tensor([0.8, 0.9])
        self.f_blood = torch.tensor([0.02, 0.03])
        self.wavelengths = np.array([500e-9, 600e-9, 700e-9])
        self.a_mie = torch.tensor([1000.0, 2000.0])
        self.b_mie = torch.tensor([1.5, 2.0])

    def test_compute_mu_a_valid(self) -> None:
        """Test valid total absorption computation."""
        result = compute_mu_a(self.sao2, self.f_blood, self.wavelengths)

        assert result.shape == (2, 3)
        assert torch.all(result >= 0)

    def test_compute_mu_a_invalid_sao2_range(self) -> None:
        """Test error with invalid sao2 range."""
        sao2 = self.sao2.clone()
        sao2[0] = 1.5  # > 1.0

        with pytest.raises(ValueError, match="sao2 must be in"):
            compute_mu_a(sao2, self.f_blood, self.wavelengths)

        sao2 = self.sao2.clone()
        sao2[0] = -0.1  # < 0.0

        with pytest.raises(ValueError, match="sao2 must be in"):
            compute_mu_a(sao2, self.f_blood, self.wavelengths)

    def test_compute_mu_a_invalid_f_blood_range(self) -> None:
        """Test error with invalid f_blood range."""
        f_blood = self.f_blood.clone()
        f_blood[0] = 1.5  # > 1.0

        with pytest.raises(ValueError, match="f_blood must be in"):
            compute_mu_a(self.sao2, f_blood, self.wavelengths)

        f_blood = self.f_blood.clone()
        f_blood[0] = -0.1  # < 0.0

        with pytest.raises(ValueError, match="f_blood must be in"):
            compute_mu_a(self.sao2, f_blood, self.wavelengths)

    def test_compute_mu_a_invalid_dimensions(self) -> None:
        """Test error with invalid input dimensions."""
        sao2 = self.sao2.clone().unsqueeze(0)

        with pytest.raises(ValueError, match=r"All inputs must be 1D."):
            compute_mu_a(sao2, self.f_blood, self.wavelengths)

        f_blood = self.f_blood.clone().unsqueeze(0)

        with pytest.raises(ValueError, match=r"All inputs must be 1D."):
            compute_mu_a(self.sao2, f_blood, self.wavelengths)

        wavelengths = copy.deepcopy(self.wavelengths)[None, :]

        with pytest.raises(ValueError, match=r"All inputs must be 1D."):
            compute_mu_a(self.sao2, self.f_blood, wavelengths)

    def test_compute_mu_a_input_type_transformation(self) -> None:
        """Test compute_mu_a with numpy inputs."""
        result = compute_mu_a(self.sao2.numpy(), self.f_blood, self.wavelengths)

        assert result.shape == (2, 3)
        assert torch.all(result >= 0)

        result = compute_mu_a(self.sao2, self.f_blood.numpy(), self.wavelengths)

        assert result.shape == (2, 3)
        assert torch.all(result >= 0)

        result = compute_mu_a(
            self.sao2, self.f_blood, torch.from_numpy(self.wavelengths)
        )

        assert result.shape == (2, 3)
        assert torch.all(result >= 0)

    def test_compute_mu_s_prime_valid(self) -> None:
        """Test valid reduced scattering computation."""
        result = compute_mu_s_prime(self.a_mie, self.b_mie, self.wavelengths)

        assert result.shape == (2, 3)
        assert torch.all(result >= 0)

    def test_compute_mu_s_prime_invalid_a_mie(self) -> None:
        """Test error with invalid a_mie values."""
        a_mie = self.a_mie.clone()
        a_mie[0] = -100.0  # Negative

        with pytest.raises(ValueError, match="a_mie must be positive"):
            compute_mu_s_prime(a_mie, self.b_mie, self.wavelengths)

    def test_compute_mu_s_prime_invalid_b_mie(self) -> None:
        """Test error with invalid b_mie values."""
        b_mie = self.b_mie.clone()
        b_mie[0] = -1.0  # Negative

        with pytest.raises(ValueError, match="b_mie must be positive"):
            compute_mu_s_prime(self.a_mie, b_mie, self.wavelengths)

    def test_compute_mu_s_prime_invalid_dimensions(self) -> None:
        """Test error with invalid input dimensions."""
        a_mie = self.a_mie.clone().unsqueeze(0)

        with pytest.raises(ValueError, match="All inputs must be 1D"):
            compute_mu_s_prime(a_mie, self.b_mie, self.wavelengths)

        b_mie = self.b_mie.clone().unsqueeze(0)

        with pytest.raises(ValueError, match="All inputs must be 1D"):
            compute_mu_s_prime(self.a_mie, b_mie, self.wavelengths)

        wavelengths = copy.deepcopy(self.wavelengths)[None, :]

        with pytest.raises(ValueError, match="All inputs must be 1D"):
            compute_mu_s_prime(self.a_mie, self.b_mie, wavelengths)

    def test_compute_mu_s_prime_input_type_transformation(self) -> None:
        """Test compute_mu_s_prime with numpy inputs."""
        result = compute_mu_s_prime(
            self.a_mie.numpy(), self.b_mie.numpy(), self.wavelengths
        )

        assert result.shape == (2, 3)
        assert torch.all(result >= 0)

        result = compute_mu_s_prime(self.a_mie, self.b_mie.numpy(), self.wavelengths)

        assert result.shape == (2, 3)
        assert torch.all(result >= 0)

        result = compute_mu_s_prime(
            self.a_mie, self.b_mie, torch.from_numpy(self.wavelengths)
        )

        assert result.shape == (2, 3)
        assert torch.all(result >= 0)


class TestReflectanceModels:
    """Test cases for reflectance model functions."""

    def setup_method(self) -> None:
        self.mu_a = torch.tensor([[1.0, 2.0, 3.0]])
        self.mu_s_prime = torch.tensor([[10.0, 20.0, 30.0]])
        self.params = {"M1": 24.0, "M2": -0.6, "M3": 56.0}
        self.params_jacques = {"M1": 7.0, "M2": 0.1, "M3": 2.0}

    def test_modified_beer_lambert_valid(self) -> None:
        """Test valid modified Beer-Lambert computation."""
        result = modified_beer_lambert(self.mu_a, self.mu_s_prime, self.params)

        assert result.shape == (1, 3)
        assert torch.all(result >= 0) and torch.all(result <= 1)

    def test_modified_beer_lambert_numpy_inputs(self) -> None:
        """Test modified Beer-Lambert with numpy inputs."""
        result = modified_beer_lambert(
            self.mu_a.numpy(), self.mu_s_prime.numpy(), self.params
        )

        assert result.shape == (1, 3)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_modified_beer_lambert_mixed_types(self) -> None:
        """Test error with mixed input types."""
        with pytest.raises(TypeError, match="mu_a and mu_s_prime must be of same type"):
            modified_beer_lambert(self.mu_a.numpy(), self.mu_s_prime, self.params)

        with pytest.raises(TypeError, match="mu_a and mu_s_prime must be of same type"):
            modified_beer_lambert(self.mu_a, self.mu_s_prime.numpy(), self.params)

    def test_modified_beer_lambert_negative_coefficients(self) -> None:
        """Test error with negative optical coefficients."""
        mu_a = self.mu_a.clone()
        mu_a[0, 0] = -1.0  # Negative

        with pytest.raises(ValueError, match="Optical coefficients must be positive"):
            modified_beer_lambert(mu_a, self.mu_s_prime, self.params)

        with pytest.raises(ValueError, match="Optical coefficients must be positive"):
            modified_beer_lambert(mu_a.numpy(), self.mu_s_prime.numpy(), self.params)

        mu_s_prime = self.mu_s_prime.clone()
        mu_s_prime[0, 0] = -10.0  # Negative

        with pytest.raises(ValueError, match="Optical coefficients must be positive"):
            modified_beer_lambert(self.mu_a, mu_s_prime, self.params)

        with pytest.raises(ValueError, match="Optical coefficients must be positive"):
            modified_beer_lambert(self.mu_a.numpy(), mu_s_prime.numpy(), self.params)

    def test_modified_beer_lambert_zero_coefficients(self) -> None:
        """Test error with zero optical coefficients."""
        mu_a = self.mu_a.clone()
        mu_a[0, 0] = 0.0  # Zero

        with pytest.raises(ValueError, match="Optical coefficients must be positive"):
            modified_beer_lambert(mu_a, self.mu_s_prime, self.params)

        with pytest.raises(ValueError, match="Optical coefficients must be positive"):
            modified_beer_lambert(mu_a.numpy(), self.mu_s_prime.numpy(), self.params)

        mu_s_prime = self.mu_s_prime.clone()
        mu_s_prime[0, 0] = 0.0  # Zero

        with pytest.raises(ValueError, match="Optical coefficients must be positive"):
            modified_beer_lambert(self.mu_a, mu_s_prime, self.params)

        with pytest.raises(ValueError, match="Optical coefficients must be positive"):
            modified_beer_lambert(self.mu_a.numpy(), mu_s_prime.numpy(), self.params)

    def test_modified_beer_lambert_edge_cases(self) -> None:
        """Test modified Beer-Lambert for edge case parameters."""
        mu_a = np.array([[1e-8, 1e-8, 100000.0, 100000.0]])
        mu_s_prime = np.array([[1e-8, 500000.0, 1e-8, 500000.0]])

        result = modified_beer_lambert(mu_a, mu_s_prime, self.params)

        assert result.shape == (1, 4)
        # NOTE: Near zero absorption cases creates non-sensical values
        assert np.allclose(result[0, 2:], [0.0, 0.0])

    def test_jacques_1999_valid(self) -> None:
        """Test valid Jacques 1999 computation."""
        result = jacques_1999(self.mu_a, self.mu_s_prime, self.params_jacques)

        assert result.shape == (1, 3)
        assert torch.all(result >= 0) and torch.all(result <= 1)

        result = jacques_1999(
            self.mu_a.numpy(), self.mu_s_prime.numpy(), self.params_jacques
        )

        assert result.shape == (1, 3)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_jacques_1999_mixed_types(self) -> None:
        """Test error with mixed input types."""
        with pytest.raises(TypeError, match="mu_a and mu_s_prime must be of same type"):
            jacques_1999(self.mu_a, self.mu_s_prime.numpy(), self.params_jacques)

        with pytest.raises(TypeError, match="mu_a and mu_s_prime must be of same type"):
            jacques_1999(self.mu_a.numpy(), self.mu_s_prime, self.params_jacques)

    def test_jacques_1999_non_positive_coefficients(self) -> None:
        """Test error with non-positive optical coefficients."""
        mu_a = self.mu_a.clone()
        mu_a[0, 0] = 0.0  # Zero absorption

        with pytest.raises(ValueError, match="Optical coefficients must be positive"):
            jacques_1999(mu_a, self.mu_s_prime, self.params_jacques)

        mu_a = self.mu_a.clone()
        mu_a[0, 0] = -1.0  # Negative absorption

        with pytest.raises(ValueError, match="Optical coefficients must be positive"):
            jacques_1999(mu_a, self.mu_s_prime, self.params_jacques)

        mu_s_prime = self.mu_s_prime.clone()
        mu_s_prime[0, 0] = 0.0  # Zero scattering

        with pytest.raises(ValueError, match="Optical coefficients must be positive"):
            jacques_1999(self.mu_a, mu_s_prime, self.params_jacques)

        mu_s_prime = self.mu_s_prime.clone()
        mu_s_prime[0, 0] = -10.0  # Negative scattering

        with pytest.raises(ValueError, match="Optical coefficients must be positive"):
            jacques_1999(self.mu_a, mu_s_prime, self.params_jacques)


class TestAnalyticalModelConstants:
    """Test cases for AnalyticalModelConstants."""

    def test_constants_default_values(self) -> None:
        """Test default constant values."""
        constants = AnalyticalModelConstants()

        assert constants.n_sims == 70000
        assert constants.n_photons == 1000000
        assert constants.thickness == 0.03
        assert constants.wavelengths.shape == (351,)
        assert constants.sao2_range == (0.0, 1.0)
        assert constants.f_blood_range == (0.002, 0.07)
        assert constants.a_mie_range == (800.0, 7000.0)
        assert constants.b_mie_range == (0.1, 3.3)
        assert constants.g_range == (0.7, 0.9)

    def test_constants_modified_beer_lambert_params(self) -> None:
        """Test modified Beer-Lambert parameters."""
        constants = AnalyticalModelConstants()

        # Check that all refractive indices are present
        assert 1.33 in constants.modified_beer_lambert_params
        assert 1.35 in constants.modified_beer_lambert_params
        assert 1.44 in constants.modified_beer_lambert_params

        # Check parameter structure
        params_1_33 = constants.modified_beer_lambert_params[1.33]
        assert "M1" in params_1_33
        assert "M2" in params_1_33
        assert "M3" in params_1_33

        params_1_35 = constants.modified_beer_lambert_params[1.35]
        assert "M1" in params_1_35
        assert "M2" in params_1_35
        assert "M3" in params_1_35

        params_1_44 = constants.modified_beer_lambert_params[1.44]
        assert "M1" in params_1_44
        assert "M2" in params_1_44
        assert "M3" in params_1_44

    def test_constants_jacques_1999_params(self) -> None:
        """Test Jacques 1999 parameters."""
        constants = AnalyticalModelConstants()

        # Check that all refractive indices are present
        assert 1.33 in constants.jacques_1999_params
        assert 1.35 in constants.jacques_1999_params
        assert 1.44 in constants.jacques_1999_params

        # Check parameter structure
        params_1_33 = constants.jacques_1999_params[1.33]
        assert "M1" in params_1_33
        assert "M2" in params_1_33
        assert "M3" in params_1_33

        params_1_35 = constants.jacques_1999_params[1.35]
        assert "M1" in params_1_35
        assert "M2" in params_1_35
        assert "M3" in params_1_35

        params_1_44 = constants.jacques_1999_params[1.44]
        assert "M1" in params_1_44
        assert "M2" in params_1_44
        assert "M3" in params_1_44

    def test_constants_jacques_1999_params_specular(self) -> None:
        """Test Jacques 1999 specular parameters."""
        constants = AnalyticalModelConstants()

        # Check that all refractive indices are present
        assert 1.33 in constants.jacques_1999_params_specular
        assert 1.35 in constants.jacques_1999_params_specular
        assert 1.44 in constants.jacques_1999_params_specular

        # Check parameter structure
        params_1_33 = constants.jacques_1999_params_specular[1.33]
        assert "M1" in params_1_33
        assert "M2" in params_1_33
        assert "M3" in params_1_33

        params_1_35 = constants.jacques_1999_params_specular[1.35]
        assert "M1" in params_1_35
        assert "M2" in params_1_35
        assert "M3" in params_1_35

        params_1_44 = constants.jacques_1999_params_specular[1.44]
        assert "M1" in params_1_44
        assert "M2" in params_1_44
        assert "M3" in params_1_44
