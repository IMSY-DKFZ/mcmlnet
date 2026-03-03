"""Utility functions to load haemoglobin extinction coefficients."""

import os

import numpy as np
from dotenv import load_dotenv
from scipy.interpolate import interp1d

from mcmlnet.constants import (
    CM_INV_TO_M_INV,
    NM_TO_M,
)

load_dotenv()


def get_haemoglobin_extinction_coefficients(
    reference_filename: str | None = None,
) -> tuple[interp1d, interp1d, tuple[float, float]]:
    """Load haemoglobin extinction coefficients from Scott Prahl's reference data.

    Reference: https://omlc.org/spectra/hemoglobin/summary.html

    The function loads extinction coefficients for oxyhemoglobin (HbO2) and
    deoxyhemoglobin (Hb) and converts units from [nm, cm^-1/(moles/l)] to
    [m, m^-1/(moles/l)] for consistency with SI units.

    Args:
        reference_filename: Path to file with extinction coefficients.
            If None, uses default path from environment.

    Returns:
        Tuple containing:
            - eHbO2: Interpolation function for oxyhemoglobin extinction coefficients
            - eHb: Interpolation function for deoxyhemoglobin extinction coefficients
            - bounds: (min_wavelength, max_wavelength) in meters

    Raises:
        FileNotFoundError: If the reference file does not exist.
        ValueError: If the data format is invalid or if wavelength range is unrealistic.

    """
    if reference_filename is None:
        reference_filename = os.path.join(
            os.environ["data_dir"], "chromophores/haemoglobin.txt"
        )

    if not os.path.exists(reference_filename):
        raise FileNotFoundError(
            f"Haemoglobin data file not found: {reference_filename}"
        )

    try:
        hemo_lut = np.loadtxt(reference_filename, skiprows=2)
    except (OSError, ValueError) as err:
        raise ValueError(
            f"Failed to load haemoglobin data from {reference_filename}: {err}"
        ) from err

    # Convert units: wavelength from nm to m, extinction coefficients from cm^-1 to m^-1
    wavelengths = hemo_lut[:, 0] * NM_TO_M
    hbo2_extinction = hemo_lut[:, 1] * CM_INV_TO_M_INV
    hb_extinction = hemo_lut[:, 2] * CM_INV_TO_M_INV

    # Validate wavelength range (should be reasonable for visible/NIR light)
    if wavelengths.min() <= 0 or wavelengths.max() > 5e-6:  # 0 to 5000 nm
        raise ValueError(
            f"Unrealistic wavelength range: {wavelengths.min() * 1e9:.1f} - "
            f"{wavelengths.max() * 1e9:.1f} nm"
        )
    # Check for monotonic wavelength ordering
    if not np.all(np.diff(wavelengths) > 0):
        raise ValueError("Wavelengths must be in strictly increasing order")

    # Create interpolation functions with extrapolation disabled for safety
    e_hbo2 = interp1d(
        wavelengths,
        hbo2_extinction,
        bounds_error=True,
    )
    e_hb = interp1d(
        wavelengths,
        hb_extinction,
        bounds_error=True,
    )
    bounds = (wavelengths.min(), wavelengths.max())

    return e_hbo2, e_hb, bounds
