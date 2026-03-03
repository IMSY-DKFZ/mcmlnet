"""
Common place to define analytical baselines for the supersampling problem.

This module provides building blocks and complete analytical models for
computing optical reflectance spectra from physical parameters.

Sources:
- https://arxiv.org/pdf/2312.12935.pdf
- https://omlc.org/news/may99/rd/index.html

More Interpolation Sources:
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html#scipy.interpolate.LinearNDInterpolator
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html#scipy.interpolate.griddata
- https://geostat-framework.readthedocs.io/projects/gstools/en/stable/#kriging-and-conditioned-random-fields
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from mcmlnet.constants import (
    NM_TO_M,
    REFERENCE_WAVELENGTH_M,
)
from mcmlnet.utils.haemoglobin_extinctions import (
    get_haemoglobin_extinction_coefficients,
)
from mcmlnet.utils.tensor import TensorType


def _mu_a_blood(
    sao2: torch.Tensor,
    wavelengths: np.ndarray,
    chb: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the absorption coefficient of the blood.

    See: https://doi.org/10.1002/jbio.202300536

    Args:
        sao2: Oxygen saturation. Range: [0,1] - in paper: 0%-100%
        wavelengths: Wavelengths [m]
        chb: Haemoglobin concentration [g/L] - default is 150 g/L

    Returns:
        Absorption coefficient [m^-1]

    Raises:
        ValueError: If inputs have invalid dimensions
    """
    # Adapt the shape of the inputs
    sao2 = sao2[:, None]
    wavelengths = wavelengths[np.newaxis, :]
    if chb is None:
        chb = torch.tensor([[150.0]])
    if not (sao2.ndim == wavelengths.ndim == chb.ndim == 2):
        raise ValueError("All inputs must be 2D after reshaping")

    # Get haemoglobin extinction coefficients
    eHbO2, eHb, _ = get_haemoglobin_extinction_coefficients()
    eHbO2 = torch.from_numpy(eHbO2(wavelengths))
    eHb = torch.from_numpy(eHb(wavelengths))

    return (
        chb * torch.log(torch.tensor([10])) / 64500 * (eHbO2 * sao2 + eHb * (1 - sao2))
    )


def _mu_a_background(wavelengths: np.ndarray) -> torch.Tensor:
    """
    Compute the background (non-hemoglobin) absorption coefficient of tissue.

    Based on Eq. 3 from https://doi.org/10.1002/jbio.202300536

    Note on units: The original equation does not specify units for the empirical
    coefficient (7.84 x 10^8). We assume the result is in m^-1, which may overestimate
    background absorption by a factor of 100x compared to literature values  in cm^-1.
    However, this assumption has minimal practical impact because:

    1. Background absorption remains small (~1 m^-1 at 500nm, ~0.1 m^-1 at 1000nm)
    2. Effect is primarily limited to tissues with very low blood volume fraction
    3. Hemoglobin absorption dominates in almost all physiologically relevant scenarios

    Args:
        wavelengths: Wavelengths [m]

    Returns:
        Absorption coefficient [m^-1]
    """
    # Convert to nm
    _wavelengths = wavelengths / NM_TO_M
    return torch.from_numpy(7.84 * 10**8 * _wavelengths**-3.255)[None, :]


def compute_mu_a(
    sao2: TensorType,
    f_blood: TensorType,
    wavelengths: TensorType,
) -> torch.Tensor:
    """Compute the total absorption coefficient as defined in Bahl et al.

    - https://doi.org/10.1002/jbio.202300536
    - https://arxiv.org/abs/2312.12935

    Args:
        sao2: Oxygen saturation. Range: [0,1] - in paper: 0%-100%
        f_blood: Blood volume fraction. Range: [0,1] - in paper: 0.2%-7%
        wavelengths: Wavelengths [m]

    Returns:
        Absorption coefficient [m^-1]

    Raises:
        ValueError: If inputs have invalid dimensions or values
    """
    # Validate parameter ranges
    if not (0 <= sao2.min() and sao2.max() <= 1):
        raise ValueError(
            f"sao2 must be in [0,1], got range [{sao2.min():.3f}, {sao2.max():.3f}]"
        )
    if not (0 <= f_blood.min() and f_blood.max() <= 1):
        raise ValueError(
            "f_blood must be in [0,1], got range "
            f"[{f_blood.min():.3f}, {f_blood.max():.3f}]"
        )

    # Ensure tensor compatibility
    if isinstance(wavelengths, torch.Tensor):
        wavelengths = wavelengths.detach().cpu().numpy()
    if isinstance(sao2, np.ndarray):
        sao2 = torch.from_numpy(sao2)
    if isinstance(f_blood, np.ndarray):
        f_blood = torch.from_numpy(f_blood)
    if not (sao2.ndim == f_blood.ndim == wavelengths.ndim == 1):
        raise ValueError("All inputs must be 1D.")

    mu_a_blood = _mu_a_blood(sao2, wavelengths)
    mu_a_bg = _mu_a_background(wavelengths)

    return mu_a_blood * f_blood[:, None] + mu_a_bg * (1 - f_blood[:, None])


def compute_mu_s_prime(
    a_mie: TensorType,
    b_mie: TensorType,
    wavelengths: TensorType,
) -> torch.Tensor:
    """Compute the reduced scattering coefficient as defined in Bahl et al.

    See: https://doi.org/10.1002/jbio.202300536

    Args:
        a_mie: Mie scattering coefficient [m^-1] - in paper 800-7000 m^-1
        b_mie: Mie scattering power
        wavelengths: Wavelengths [m]

    Returns:
        Reduced scattering coefficient [m^-1]

    Raises:
        ValueError: If inputs have invalid dimensions or values
    """
    # Validate parameter ranges
    if a_mie.min() <= 0:
        raise ValueError(f"a_mie must be positive, got minimum {a_mie.min():.3f}")
    if b_mie.min() <= 0:
        raise ValueError(f"b_mie must be positive, got minimum {b_mie.min():.3f}")

    # Ensure tensor compatibility
    if isinstance(wavelengths, np.ndarray):
        wavelengths = torch.from_numpy(wavelengths)
    if isinstance(a_mie, np.ndarray):
        a_mie = torch.from_numpy(a_mie)
    if isinstance(b_mie, np.ndarray):
        b_mie = torch.from_numpy(b_mie)
    if not (a_mie.ndim == b_mie.ndim == wavelengths.ndim == 1):
        raise ValueError("All inputs must be 1D.")

    return (
        a_mie[:, None]
        * (wavelengths[None, :] / REFERENCE_WAVELENGTH_M) ** -b_mie[:, None]
    )


def modified_beer_lambert(
    mu_a: TensorType,
    mu_s_prime: TensorType,
    params: dict[str, float],
) -> TensorType:
    """Modified Beer-Lambert law for the calculation of the diffuse reflectance.

    Requires empirical parameters for the pathlength, scattering scaling and offset.
    See: https://doi.org/10.1002/jbio.202300536

    Args:
        mu_a: Absorption coefficient [m^-1]
        mu_s_prime: Reduced scattering coefficient [m^-1]
        params: Empirical parameters for the modified Beer-Lambert law.

    Returns:
        Diffuse reflectance [unitless, not percent]

    Raises:
        ValueError: If inputs are invalid
        TypeError: If mu_a and mu_s_prime are different types
    """
    # Validate inputs are same type
    if type(mu_a) is not type(mu_s_prime):
        raise TypeError("mu_a and mu_s_prime must be of same type")

    # Validate input ranges
    if isinstance(mu_a, torch.Tensor):
        if (mu_a <= 0).any() or (mu_s_prime <= 0).any():
            raise ValueError("Optical coefficients must be positive")
    elif isinstance(mu_a, np.ndarray):
        if (mu_a <= 0).any() or (mu_s_prime <= 0).any():
            raise ValueError("Optical coefficients must be positive")

    # Convert to cm^-1 (factor of 100 conversion from m^-1)
    mu_a_cm = mu_a / 100
    mu_s_prime_cm = mu_s_prime / 100

    # Calculate empirical absorbance
    empirical_absorbance = (
        params["M1"] * mu_a_cm + params["M2"] * mu_s_prime_cm + params["M3"]
    )

    # NOTE: Paper mentions rescaling from mm^-1 to cm^-1,
    # but uses factor of 100 instead of 10 for conversion ...?
    if isinstance(mu_a, torch.Tensor):
        diffuse_reflectance = torch.exp(-empirical_absorbance / 100)
    else:
        diffuse_reflectance = np.exp(-empirical_absorbance / 100)

    return diffuse_reflectance


def jacques_1999(
    mu_a: TensorType,
    mu_s_prime: TensorType,
    params: dict[str, float],
) -> TensorType:
    """Jacques' 1999 approximation of the diffuse reflectance.

    Given the absorption and reduced scattering parameters.
    - https://omlc.org/news/may99/rd/index.html

    Args:
        mu_a: Absorption coefficient [m^-1]
        mu_s_prime: Reduced scattering coefficient [m^-1]
        params: Fitting parameters

    Returns:
        Diffuse reflectance [unitless, not percent]

    Raises:
        ValueError: If inputs are invalid
        TypeError: If mu_a and mu_s_prime are different types
    """
    # Validate inputs are same type
    if type(mu_a) is not type(mu_s_prime):
        raise TypeError("mu_a and mu_s_prime must be of same type")

    # Validate input ranges
    if isinstance(mu_a, torch.Tensor):
        if (mu_a <= 0).any() or (mu_s_prime <= 0).any():
            raise ValueError("Optical coefficients must be positive for Jacques model")
    elif isinstance(mu_a, np.ndarray):
        if (mu_a <= 0).any() or (mu_s_prime <= 0).any():
            raise ValueError("Optical coefficients must be positive for Jacques model")

    # Convert to cm^-1
    mu_a = mu_a / 100
    mu_s_prime = mu_s_prime / 100

    # Calculate parameter quotient, pathlength and absorbance
    quotient = mu_s_prime / mu_a
    if isinstance(mu_a, torch.Tensor):
        delta_lambda = 1.0 / torch.sqrt(3 * mu_a * (mu_a + mu_s_prime))
        absorbance = params["M1"] + params["M2"] * torch.exp(
            torch.log(quotient) / params["M3"]
        )
        diffuse_reflectance = torch.exp(-absorbance * delta_lambda * mu_a)
    else:
        delta_lambda = 1.0 / np.sqrt(3 * mu_a * (mu_a + mu_s_prime))
        absorbance = params["M1"] + params["M2"] * np.exp(
            np.log(quotient) / params["M3"]
        )
        diffuse_reflectance = np.exp(-absorbance * delta_lambda * mu_a)

    return diffuse_reflectance


def yudovsky_2009(
    mu_a: TensorType,
    mu_s_prime: TensorType,
    params: dict[str, float],
    kind: str = "original",
) -> TensorType:
    """Yudovsky et al.'s 2009 approximation of the diffuse reflectance.

    Given the absorption and reduced scattering parameters.
    - paper: https://doi.org/10.1364/AO.48.006670
    - erratum: https://doi.org/10.1364/ao.54.006116

    Args:
        mu_a: Absorption coefficient [m^-1]
        mu_s_prime: Reduced scattering coefficient [m^-1]
        params: Fitting parameters
        kind: Kind of reported fitting function to use

    Returns:
        Diffuse reflectance [unitless, not percent]

    Raises:
        ValueError: If inputs are invalid
        TypeError: If mu_a and mu_s_prime are different types
    """
    # Validate kind parameter
    valid_kinds = ["original", "erratum", "bahl_reproduced"]
    if kind not in valid_kinds:
        raise ValueError(f"kind must be in {valid_kinds}, got '{kind}'")

    # Validate inputs are same type
    if type(mu_a) is not type(mu_s_prime):
        raise TypeError("mu_a and mu_s_prime must be of same type")

    # Validate input ranges
    if isinstance(mu_a, torch.Tensor):
        if (mu_a <= 0).any() or (mu_s_prime <= 0).any():
            raise ValueError("Optical coefficients must be positive for Yudovsky model")
    elif isinstance(mu_a, np.ndarray):
        if (mu_a <= 0).any() or (mu_s_prime <= 0).any():
            raise ValueError("Optical coefficients must be positive for Yudovsky model")

    # Convert to cm^-1
    mu_a = mu_a / 100
    mu_s_prime = mu_s_prime / 100

    # Calculate reduced albedo
    reduced_albedo = mu_s_prime / (mu_a + mu_s_prime)

    # NOTE: Error in Bahl et al.? M6 should be omega' (compare to Erratum by Yudovsky)
    if kind == "original":
        denominator = 1 - params["M6"] * reduced_albedo
    elif kind == "erratum":
        denominator = params["M6"] - reduced_albedo
    else:
        denominator = 1.02 - params["M6"]

    if isinstance(mu_a, torch.Tensor):
        diffuse_reflectance = (
            params["M1"]
            + params["M2"] * torch.exp(params["M3"] * reduced_albedo ** params["M4"])
            + params["M5"] / denominator
        )
    else:
        diffuse_reflectance = (
            params["M1"]
            + params["M2"] * np.exp(params["M3"] * reduced_albedo ** params["M4"])
            + params["M5"] / denominator
        )

    return diffuse_reflectance


@dataclass(frozen=True)
class AnalyticalModelConstants:
    """Constants and parameters for analytical surrogate models.

    Based on Bahl et al. (https://doi.org/10.1002/jbio.202300536):

    Simulation ranges:
    - sao2: 0-100% oxygen saturation
    - f_blood: 0.2-7% blood volume fraction
    - a_mie: 800-7000 m^-1 Mie scattering amplitude
    - b_mie: 0.1-3.3 Mie scattering power
    - g: 0.7-0.9 anisotropy factor
    - n: 1.33, 1.35, 1.44 refractive indices
    - d: 3 cm = 0.03 m thickness

    Each configuration used 100,000 photons across 100 simulations in the original work.
    We increase this to 1,000,000 photons per simulation and 70,000 simulations.
    """

    # Simulation parameters
    n_sims: int = field(default=70000)
    n_photons: int = field(default=1000000)
    thickness: float = field(default=0.03)  # meters

    # Wavelength range: 300-1000 nm in 2 nm steps
    wavelengths: np.ndarray = field(
        default_factory=lambda: np.linspace(
            300, 1000, 351, endpoint=True, dtype=np.float64
        )
        * 1e-9
    )

    # Parameter ranges for validation
    sao2_range: tuple[float, float] = field(default=(0.0, 1.0))
    f_blood_range: tuple[float, float] = field(default=(0.002, 0.07))
    a_mie_range: tuple[float, float] = field(default=(800.0, 7000.0))  # m^-1
    b_mie_range: tuple[float, float] = field(default=(0.1, 3.3))
    g_range: tuple[float, float] = field(default=(0.7, 0.9))

    # Model parameters (existing dictionaries)
    modified_beer_lambert_params: dict[float, dict[str, float]] = field(
        default_factory=lambda: {
            1.33: {"M1": 24.1184, "M2": -0.5866, "M3": 56.3167},
            1.35: {"M1": 24.9746, "M2": -0.5965, "M3": 57.535},
            1.44: {"M1": 29.2922, "M2": -0.6429, "M3": 63.1409},
        }
    )
    # NOTE: As visible from the parameters, the diffuse reflectance fit of the
    # Jacques 1999 model is more robust, however, has a worse fitting error.
    jacques_1999_params: dict[float, dict[str, float]] = field(
        default_factory=lambda: {
            1.33: {"M1": 7.2559, "M2": 0.0718, "M3": 2.0784},
            1.35: {"M1": 7.3239, "M2": 0.107, "M3": 2.2614},
            1.44: {"M1": 7.4669, "M2": 0.3829, "M3": 3.0265},  # lowest error
        }
    )
    jacques_1999_params_specular: dict[float, dict[str, float]] = field(
        default_factory=lambda: {
            1.33: {"M1": 6.1582, "M2": 0.5957, "M3": 6.3073},
            1.35: {"M1": 5.4329, "M2": 1.3188, "M3": 9.6161},
            1.44: {
                "M1": -16737.979,
                "M2": 16744.6398,
                "M3": 46726.4906,
            },  # lowest error
        }
    )
    yudovsky_2009_params: dict[float, dict[str, float]] = field(
        default_factory=lambda: {
            1.33: {
                "M1": -0.0908,
                "M2": 0.0344,
                "M3": 2.0783,
                "M4": 80.0256,
                "M5": 0.0559,
                "M6": 0.9256,
            },
            1.35: {
                "M1": -0.0887,
                "M2": 0.0345,
                "M3": 2.0822,
                "M4": 81.6818,
                "M5": 0.0508,
                "M6": 0.9277,
            },
            1.44: {
                "M1": -0.0802,
                "M2": 0.0353,
                "M3": 2.1012,
                "M4": 89.2555,
                "M5": 0.0424,
                "M6": 0.9362,
            },
        }
    )
