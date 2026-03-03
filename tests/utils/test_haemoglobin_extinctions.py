"""Tests for mcmlnet.utils.haemoglobin_extinctions module."""

import os
import tempfile

import numpy as np
import pytest

from mcmlnet.utils.haemoglobin_extinctions import (
    get_haemoglobin_extinction_coefficients,
)


class TestGetHaemoglobinExtinctionCoefficients:
    def test_file_is_found(self) -> None:
        """Test that the default file is found and loaded."""
        interp1, interp2, bounds = get_haemoglobin_extinction_coefficients()
        assert callable(interp1)
        assert callable(interp2)
        assert isinstance(bounds, tuple)
        assert bounds[0] < bounds[1]

    def test_file_not_found(self) -> None:
        """Test with a non-existent file path."""
        with pytest.raises(FileNotFoundError):
            get_haemoglobin_extinction_coefficients("/not/a/real/file.txt")

    def test_invalid_wavelength_range(self) -> None:
        """Test with a file where wavelengths are out of realistic range."""
        arr = np.array([[0, 1, 2], [6e3, 3, 4]])  # 0 and 6000 nm (out of range)
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            # Write 2 header lines to skip
            f.write("# header\n" * 2)
            np.savetxt(f, arr)
            fname = f.name
        try:
            with pytest.raises(ValueError, match="Unrealistic wavelength range"):
                get_haemoglobin_extinction_coefficients(fname)
        finally:
            os.remove(fname)

    def test_non_monotonic_wavelengths(self) -> None:
        """Test with a file where wavelengths are not strictly increasing."""
        arr = np.array([[500, 1, 2], [400, 3, 4]])  # Not increasing
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            # Write 2 header lines to skip
            f.write("# header\n" * 2)
            np.savetxt(f, arr)
            fname = f.name
        try:
            with pytest.raises(
                ValueError, match="Wavelengths must be in strictly increasing order"
            ):
                get_haemoglobin_extinction_coefficients(fname)
        finally:
            os.remove(fname)

    def test_valid_file(self) -> None:
        """Test with a valid temporary file."""
        arr = np.array([[500, 1, 2], [600, 3, 4]])
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            # Write 2 header lines to skip
            f.write("# header\n" * 2)
            np.savetxt(f, arr)
            fname = f.name
        try:
            e_hbo2, e_hb, bounds = get_haemoglobin_extinction_coefficients(fname)
            x = arr[0, 0] * 1e-9
            assert bounds[0] <= x <= bounds[1]
            assert e_hbo2(x) == pytest.approx(arr[0, 1] * 1e2)
            assert e_hb(x) == pytest.approx(arr[0, 2] * 1e2)
        finally:
            os.remove(fname)
