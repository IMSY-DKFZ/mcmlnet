"""Related work transformations for optical parameter calculations."""

import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from mcmlnet.constants import (
    CM_INV_TO_M_INV,
    NM_TO_M,
    RAYLEIGH_EXPONENT,
    REFERENCE_WAVELENGTH_M,
)
from mcmlnet.utils.haemoglobin_extinctions import (
    get_haemoglobin_extinction_coefficients,
)


def validate_file_and_load(filename: str, skiprows: int, min_cols: int) -> np.ndarray:
    """Validate file existence and load data with error handling.

    Args:
        filename: Path to data file
        skiprows: Number of rows to skip
        min_cols: Minimum number of columns required

    Returns:
        Loaded data array

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If data format is invalid
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Reference file not found: {filename}")

    try:
        data = np.loadtxt(filename, skiprows=skiprows)
    except (OSError, ValueError) as err:
        raise ValueError(f"Failed to load data from {filename}: {err}") from err

    if data.shape[1] < min_cols:
        raise ValueError(
            f"Invalid data format in {filename}: "
            f"expected at least {min_cols} columns, got {data.shape[1]}"
        )

    return data


def get_mu_a_epidermis(
    reference_filename: str | None = None,
) -> tuple[interp1d, tuple[float, float]]:
    """Load Qu reference data for epidermal absorption.

    Args:
        reference_filename: Path to reference file

    Returns:
        Interpolation function and bounds
    """
    if reference_filename is None:
        reference_filename = os.path.join(
            os.environ["data_dir"], "chromophores/epidermis_qu.txt"
        )
    epi_lut = validate_file_and_load(reference_filename, skiprows=2, min_cols=2)

    # Convert units: wavelength from nm to m, absorption coefficients from cm^-1 to m^-1
    epi_lut[:, 0] = epi_lut[:, 0] * NM_TO_M
    epi_lut[:, 1] = epi_lut[:, 1] * CM_INV_TO_M_INV

    # Define interpolation with specific bound values
    mu_epi = interp1d(
        epi_lut[:, 0],
        epi_lut[:, 1],
        bounds_error=False,
        fill_value=(np.nan, epi_lut[-1, 1]),
    )
    epi_bounds = (min(epi_lut[:, 0]), max(epi_lut[:, 0]))
    return mu_epi, epi_bounds


def get_mu_a_water(
    reference_filename: str | None = None,
) -> tuple[interp1d, tuple[float, float]]:
    """Load Buiteveld reference data for water absorption.

    Args:
        reference_filename: Path to reference file

    Returns:
        Interpolation function and bounds
    """
    if reference_filename is None:
        reference_filename = os.path.join(
            os.environ["data_dir"], "chromophores/water_buiteveld.txt"
        )
    water_lut = validate_file_and_load(reference_filename, skiprows=5, min_cols=2)

    # Convert units: wavelength from nm to m
    # NOTE: Buiteveld's absorption data is already in m^-1
    water_lut[:, 0] = water_lut[:, 0] * NM_TO_M

    # Define interpolation but do not allow exceeding bounds
    mu_water = interp1d(
        water_lut[:, 0],
        water_lut[:, 1],
        bounds_error=True,
    )
    water_bounds = (min(water_lut[:, 0]), max(water_lut[:, 0]))
    return mu_water, water_bounds


def get_mu_a_collagen(
    reference_filename: str | None = None,
) -> tuple[interp1d, tuple[float, float]]:
    """Load Nunez reference data for collagen absorption.

    Args:
        reference_filename: Path to reference file

    Returns:
        Interpolation function and bounds
    """
    if reference_filename is None:
        reference_filename = os.path.join(
            os.environ["data_dir"], "chromophores/collagen_nunez.txt"
        )
    collagen_lut = validate_file_and_load(reference_filename, skiprows=2, min_cols=2)

    # Convert units: wavelength from nm to m, absorption coefficients from cm^-1 to m^-1
    collagen_lut[:, 0] = collagen_lut[:, 0] * NM_TO_M
    collagen_lut[:, 1] = collagen_lut[:, 1] * CM_INV_TO_M_INV

    # Define interpolation but do not allow exceeding bounds
    mu_collagen = interp1d(
        collagen_lut[:, 0],
        collagen_lut[:, 1],
        bounds_error=True,
    )
    collagen_bounds = (min(collagen_lut[:, 0]), max(collagen_lut[:, 0]))
    return mu_collagen, collagen_bounds


def get_oxidised_cytochrome_c(
    reference_filename: str | None = None,
    smooth: bool = True,
) -> tuple[interp1d, tuple[float, float]]:
    """Load Mason reference data for extinction of oxidized cytochrome c.

    Args:
        reference_filename: Path to reference file
        smooth: Whether to smooth the data

    Returns:
        Interpolation function and bounds
    """
    if reference_filename is None:
        reference_filename = os.path.join(
            os.environ["data_dir"], "chromophores/oxidised_cytochrome_c_mason.txt"
        )
    cyto_lut = validate_file_and_load(reference_filename, skiprows=2, min_cols=2)

    # Remove noisy data points beyond 970 nm
    cyto_lut = cyto_lut[cyto_lut[:, 0] < 970]
    # Convert units: wavelength from nm to m, extinction coefficients from cm^-1 to m^-1
    cyto_lut[:, 0] = cyto_lut[:, 0] * NM_TO_M
    cyto_lut[:, 1] = cyto_lut[:, 1] * CM_INV_TO_M_INV

    if smooth:
        # Smooth signal because cytochrome c is noisy beyond ~900 nm
        cyto_lut[:, 1] = savgol_filter(cyto_lut[:, 1], window_length=25, polyorder=3)

    # Define interpolation with specific bound values
    e_cyto = interp1d(
        cyto_lut[:, 0],
        cyto_lut[:, 1],
        bounds_error=False,
        fill_value=(np.nan, cyto_lut[-1, 1]),
    )
    cyto_bounds = (min(cyto_lut[:, 0]), max(cyto_lut[:, 0]))
    return e_cyto, cyto_bounds


def get_reduced_cytochrome_c(
    reference_filename: str | None = None,
    smooth: bool = True,
) -> tuple[interp1d, tuple[float, float]]:
    """Load Mason reference data for extinction of reduced cytochrome c.

    Args:
        reference_filename: Path to reference file
        smooth: Whether to smooth the data

    Returns:
        Interpolation function and bounds
    """
    if reference_filename is None:
        reference_filename = os.path.join(
            os.environ["data_dir"], "chromophores/reduced_cytochrome_c_mason.txt"
        )
    cyto_lut = validate_file_and_load(reference_filename, skiprows=2, min_cols=2)

    # Remove noisy data points beyond 970 nm
    cyto_lut = cyto_lut[cyto_lut[:, 0] < 970]
    # Convert units: wavelength from nm to m, extinction coefficients from cm^-1 to m^-1
    cyto_lut[:, 0] = cyto_lut[:, 0] * NM_TO_M
    cyto_lut[:, 1] = cyto_lut[:, 1] * CM_INV_TO_M_INV

    if smooth:
        # Smooth signal because cytochrome c is noisy beyond ~900 nm
        cyto_lut[:, 1] = savgol_filter(cyto_lut[:, 1], window_length=25, polyorder=3)

    # Define interpolation with specific bound values
    e_cyto = interp1d(
        cyto_lut[:, 0],
        cyto_lut[:, 1],
        bounds_error=False,
        fill_value=(np.nan, cyto_lut[-1, 1]),
    )
    cyto_bounds = (min(cyto_lut[:, 0]), max(cyto_lut[:, 0]))
    return e_cyto, cyto_bounds


def get_bilirubin_extended_upper_range(
    reference_filename: str | None = None,
    smooth: bool = True,
) -> tuple[interp1d, tuple[float, float]]:
    """Load bilirubin extinction coefficient interpolation function.

    Args:
        reference_filename: Path to reference file
        smooth: Whether to smooth the data

    Returns:
        Interpolation function and bounds
    """
    if reference_filename is None:
        reference_filename = os.path.join(
            os.environ["data_dir"], "chromophores/bilirubin.txt"
        )
    bili_lut = validate_file_and_load(reference_filename, skiprows=2, min_cols=2)

    # Convert units: wavelength from nm to m, extinction coefficients from cm^-1 to m^-1
    bili_lut[:, 0] = bili_lut[:, 0] * NM_TO_M
    # Clip negative extinction coefficient due to noise in the data
    bili_lut[:, 1] = np.abs(bili_lut[:, 1] * CM_INV_TO_M_INV)

    if smooth:
        # Smooth signal because bilirubin is noisy between 600 and 700 nm
        bili_lut[:, 1] = savgol_filter(bili_lut[:, 1], window_length=55, polyorder=3)

    # Define interpolation with specific bound values
    e_bili = interp1d(
        bili_lut[:, 0], bili_lut[:, 1], bounds_error=False, fill_value=(np.nan, 0)
    )
    bili_bounds = (min(bili_lut[:, 0]), max(bili_lut[:, 0]))
    return e_bili, bili_bounds


class Transformation:
    """Base class for optical parameter transformations."""

    def __init__(self, wavelengths: torch.Tensor):
        """Base class for transformations.

        Args:
            wavelengths: Wavelengths tensor (1D), in nanometers.

        Raises:
            ValueError: If wavelengths are not 1D or out of bounds.
        """
        self.wavelengths = wavelengths
        if wavelengths.ndim != 1:
            raise ValueError(
                f"Wavelengths must be a 1D tensor, got {wavelengths.ndim}D"
            )
        if not torch.all(wavelengths > 200) or not torch.all(wavelengths < 2000):
            raise ValueError(
                f"Wavelengths must be between 200 and 2000 nm, got range "
                f"[{wavelengths.min():.1f}, {wavelengths.max():.1f}]"
            )

    @staticmethod
    def validate_tensor_range(
        tensor: torch.Tensor, min_val: float, max_val: float, name: str
    ) -> None:
        """Validate that tensor values are within a specified range.

        Args:
            tensor: Tensor to validate
            min_val: Minimum valid value
            max_val: Maximum valid value
            name: Name of the tensor for logging

        Raises:
            ValueError: If tensor values are out of range
        """
        if not torch.all((tensor >= min_val) & (tensor <= max_val)):
            raise ValueError(
                f"{name} values must be in range [{min_val}, {max_val}], got "
                f"[{tensor.min():.6f}, {tensor.max():.6f}]"
            )


class TsuiTransformation(Transformation):
    """https://doi.org/10.1364/BOE.490164"""

    def __init__(self, wavelengths: torch.Tensor):
        super().__init__(wavelengths)

        # Extinction coefficients for HbO2 and Hb are loaded in m^-1 / (M * l)
        eHbO2, eHb, _ = get_haemoglobin_extinction_coefficients()
        self.eHbO2 = torch.from_numpy(eHbO2(self.wavelengths * NM_TO_M))
        self.eHb = torch.from_numpy(eHb(self.wavelengths * NM_TO_M))
        self.cHb = 150  # g/l - not explicitly stated in the paper
        self.gmw_hb = 64500  # g/mol
        self.gmw_hbo2 = 64532  # g/mol

        # Absorption coefficient of the epidermis
        mu_a_epi, _ = get_mu_a_epidermis()
        self.mu_a_epi = torch.from_numpy(mu_a_epi(self.wavelengths * NM_TO_M))

        # Melanin absorption coefficient
        self.mu_a_mel = 1.7 * 10**14 * torch.pow(self.wavelengths, -3.48)

        # Water absorption coefficient
        mu_a_water, _ = get_mu_a_water()
        self.mu_a_water = torch.from_numpy(mu_a_water(self.wavelengths * NM_TO_M))

        # Collagen absorption coefficient
        mu_a_col, _ = get_mu_a_collagen()
        self.mu_a_col = torch.from_numpy(mu_a_col(self.wavelengths * NM_TO_M))

    def _hemoglobin_total_absorption(self, sao2: torch.Tensor) -> torch.Tensor:
        """Compute the total hemoglobin absorption coefficient.

        Args:
            sao2: Oxygen saturation (0-1)

        Returns:
            Total hemoglobin absorption coefficient in m^-1

        Raises:
            ValueError: If sao2 is not 1D or out of valid range
        """
        if sao2.ndim != 1:
            raise ValueError(f"sao2 must be a 1D tensor, got {sao2.ndim}D")
        self.validate_tensor_range(sao2, 0, 1, "sao2")

        # Compute absorption coefficient in m^-1
        ua = (
            math.log(10)
            * self.cHb
            * (
                sao2[:, None] * self.eHbO2[None, :] / self.gmw_hbo2
                + (1 - sao2[:, None]) * self.eHb[None, :] / self.gmw_hb
            )
        )
        return ua.squeeze()

    def mu_a_1_and_2(self) -> torch.Tensor:
        """
        Compute the absorption coefficient for tissue layer one and two in Tsui's model.
        """
        return self.mu_a_epi

    def mu_a_3(self, f_m: torch.Tensor) -> torch.Tensor:
        """Compute the absorption coefficient for tissue layer three in Tsui's model.

        Args:
            f_m: Melanin volume fraction (0-1)

        Returns:
            Absorption coefficient for layer three in m^-1

        Raises:
            ValueError: If f_m is not 1D or out of valid range
        """
        if f_m.ndim != 1:
            raise ValueError(f"f_m must be a 1D tensor, got {f_m.ndim}D")
        self.validate_tensor_range(f_m, 0, 1, "melanin volume fraction f_m")

        # compute absorption coefficient in m^-1
        ua_a_3 = (
            f_m[:, None] * self.mu_a_mel[None, :]
            + (1 - f_m[:, None]) * self.mu_a_epi[None, :]
        )
        return ua_a_3.squeeze()

    def mu_a_4(
        self, f_hb: torch.Tensor, sao2: torch.Tensor, f_w: torch.Tensor
    ) -> torch.Tensor:
        """Compute the absorption coefficient for tissue layer four in Tsui's model.

        Args:
            f_hb: Hemoglobin volume fraction (0-1)
            sao2: Oxygen saturation (0-1)
            f_w: Water volume fraction (0-1)

        Returns:
            Absorption coefficient for layer four in m^-1

        Raises:
            ValueError: If input tensors are not 1D or out of valid range
        """
        # Input validation
        if not all(tensor.ndim == 1 for tensor in [f_hb, sao2, f_w]):
            raise ValueError("All input tensors must be 1D")

        n_samples = len(f_hb)
        if not all(len(tensor) == n_samples for tensor in [sao2, f_w]):
            raise ValueError("All input tensors must have the same length")

        self.validate_tensor_range(f_hb, 0, 1, "hemoglobin volume fraction f_hb")
        self.validate_tensor_range(sao2, 0, 1, "oxygen saturation sao2")
        self.validate_tensor_range(f_w, 0, 1, "water volume fraction f_w")

        # Check that fractions don't exceed 1 when combined
        total_fractions = f_hb + f_w
        if torch.any(total_fractions > 1):
            raise ValueError("Combined hemoglobin and water fractions cannot exceed 1")

        # water: https://apps.dtic.mil/sti/citations/ADA511354
        # collagen: https://doi.org/10.1117/12.190060
        ua_a_4 = (
            f_hb[:, None] * self._hemoglobin_total_absorption(sao2)
            + f_w[:, None] * self.mu_a_water[None, :]
            + (1 - f_hb[:, None] - f_w[:, None]) * self.mu_a_col[None, :]
        )
        return ua_a_4.squeeze()

    def scattering(
        self, a_mie: torch.Tensor, b_mie: torch.Tensor, g: torch.Tensor
    ) -> torch.Tensor:
        """Compute the scattering coefficient using Tsui's model.

        Args:
            a_mie: Mie scattering amplitude (m^-1)
            b_mie: Mie scattering power
            g: Anisotropy factor (0-1)

        Returns:
            Scattering coefficient in m^-1

        Raises:
            ValueError: If input tensors are not 1D or if g is not in (0.5, 1)
        """
        if not all(tensor.ndim == 1 for tensor in [a_mie, b_mie, g]):
            raise ValueError("All input tensors must be 1D")
        self.validate_tensor_range(g, 0.5, 1, "anisotropy factor g")

        # compute scattering coefficient in m^-1 (but wavelength expected in nm!)
        s = (
            a_mie[:, None]
            * (self.wavelengths[None, :]) ** -b_mie[:, None]
            / (1 - g[:, None])
        )
        return s.squeeze()


class LanTransformation(Transformation):
    """https://doi.org/10.1364/BOE.490164"""

    def __init__(self, wavelengths: torch.Tensor):
        """
        Transform physiological parameters to physical parameters using the Lan model.
        """
        super().__init__(wavelengths)

        # Extinction coefficients for HbO2 and Hb are loaded in m^-1 / (M * l)
        eHbO2, eHb, _ = get_haemoglobin_extinction_coefficients()
        self.eHbO2 = torch.from_numpy(eHbO2(self.wavelengths * NM_TO_M))
        self.eHb = torch.from_numpy(eHb(self.wavelengths * NM_TO_M))
        self.cHb = 150  # g/l - not explicitly stated in the paper
        self.gmw_hb = 64500  # g/mol - not explicitly stated in the paper

    def absorption(self, sao2: torch.Tensor, vhb: torch.Tensor) -> torch.Tensor:
        """Compute the hemoglobin absorption coefficient using the Lan model.

        Args:
            sao2: Oxygen saturation (0-1)
            vhb: Hemoglobin volume fraction (0-1)

        Returns:
            Absorption coefficient in m^-1

        Raises:
            ValueError: If input parameters are out of valid range
        """
        if vhb.ndim != 1:
            raise ValueError(f"vhb must be 1D tensor, got {vhb.ndim}D")
        if sao2.ndim != 1:
            raise ValueError(f"sao2 must be 1D tensor, got {sao2.ndim}D")
        if len(vhb) != len(sao2):
            raise ValueError(
                f"vhb and sao2 must have same length: {len(vhb)} != {len(sao2)}"
            )

        self.validate_tensor_range(vhb, 0, 1, "vhb")
        self.validate_tensor_range(sao2, 0, 1, "sao2")

        ua = (
            math.log(10)
            * self.cHb
            * (
                sao2[:, None] * self.eHbO2[None, :]
                + (1 - sao2[:, None]) * self.eHb[None, :]
            )
            * vhb[:, None]
            / self.gmw_hb
        )
        return ua.squeeze()

    def scattering(
        self,
        a_mie: torch.Tensor,
        b_mie: torch.Tensor,
        a_ray: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the scattering coefficient using the Lan model.

        Args:
            a_mie: Mie scattering amplitude
            b_mie: Mie scattering power
            a_ray: Rayleigh scattering amplitude
            g: Anisotropy factor (0-1)
        Returns:
            Scattering coefficient in m^-1

        Raises:
            ValueError: If input dimensions or ranges are invalid
        """
        if not all(tensor.ndim == 1 for tensor in [a_mie, b_mie, a_ray, g]):
            raise ValueError("All input tensors must be 1D")

        n_samples = len(a_mie)
        if not all(len(tensor) == n_samples for tensor in [b_mie, a_ray, g]):
            raise ValueError("All input tensors must have the same length")

        self.validate_tensor_range(g, 0, 1, "anisotropy factor g")

        # Ensure wavelengths are on the same device as inputs
        device = a_mie.device
        wavelengths = self.wavelengths.to(device)

        norm_wavelength = wavelengths[None, :] * NM_TO_M / REFERENCE_WAVELENGTH_M
        us_prime = (
            a_mie[:, None] * norm_wavelength ** -b_mie[:, None]
            + a_ray[:, None] * norm_wavelength**RAYLEIGH_EXPONENT
        )
        us = us_prime / (1 - g[:, None])
        return us.squeeze()


class ManojlovicTransformation(Transformation):
    """https://doi.org/10.1364/BOE.490164"""

    def __init__(self, wavelengths: torch.Tensor):
        super().__init__(wavelengths)

        # Extinction coefficients for HbO2 and Hb are loaded in m^-1 / (M * l)
        eHbO2, eHb, _ = get_haemoglobin_extinction_coefficients()
        self.eHbO2 = torch.from_numpy(eHbO2(self.wavelengths * NM_TO_M)).float()
        self.eHb = torch.from_numpy(eHb(self.wavelengths * NM_TO_M)).float()
        self.cHb = 150  # g/l - not explicitly stated in the paper
        self.gmw_hb = 64500  # g/mol - not explicitly stated in the paper

        # Extinction coefficients for bilirubin are loaded in m^-1 / (M)
        # NOTE: data looks similar enough to https://doi.org/10.1002/jbio.201300198
        eBrub, _ = get_bilirubin_extended_upper_range()
        self.eBrub = torch.from_numpy(eBrub(self.wavelengths * NM_TO_M)).float()

        # Cytochrome c extinction coefficients are loaded in m^-1 / (M)
        e_cyto_ox, _ = get_oxidised_cytochrome_c()
        self.e_cyto_ox = torch.from_numpy(e_cyto_ox(self.wavelengths * NM_TO_M)).float()
        e_cyto_red, _ = get_reduced_cytochrome_c()
        self.e_cyto_red = torch.from_numpy(
            e_cyto_red(self.wavelengths * NM_TO_M)
        ).float()

    def _melanin_absorption(self, f_m: torch.Tensor) -> torch.Tensor:
        """
        Compute the melanin absorption coefficient using the Manojlovic model.

        Args:
            f_m: Melanin volume fraction

        Returns:
            Melanin absorption coefficient

        Raises:
            ValueError: If melanin volume fraction is not 1D or out of valid range
        """
        if f_m.ndim != 1:
            raise ValueError(f"f_m must be 1D tensor, got {f_m.ndim}D")
        self.validate_tensor_range(f_m, 0.001, 0.05, "melanin volume fraction f_m")

        # Ensure wavelengths are on same device
        wavelengths = self.wavelengths.to(f_m.device)

        # Compute absorption coefficient in m^-1
        a_m = f_m[:, None] * 6.6 * 10**13 * torch.pow(wavelengths[None, :], -3.33)
        return a_m.squeeze()

    def _base_absorption(self) -> torch.Tensor:
        """
        Compute the base absorption coefficient using the Manojlovic model.

        Returns:
            Base absorption coefficient
        """
        # Compute absorption coefficient in m^-1
        a_b = 24.4 + 8530 * torch.exp(-(self.wavelengths - 154) / 66.2)
        return a_b.squeeze()

    def _hemoglobin_absorption(self, f_hb: torch.Tensor) -> torch.Tensor:
        """
        Compute the hemoglobin absorption coefficient using the Manojlovic model.

        Args:
            f_hb: Hemoglobin volume fraction

        Returns:
            Hemoglobin absorption coefficient

        Raises:
            ValueError: If hemoglobin volume fraction is not 1D or out of valid range
        """
        if f_hb.ndim != 1:
            raise ValueError(f"f_hb must be 1D tensor, got {f_hb.ndim}D")
        self.validate_tensor_range(f_hb, 0.001, 0.05, "hemoglobin volume fraction f_hb")

        # Compute absorption coefficient in m^-1
        ua = math.log(10) * self.cHb * f_hb[:, None] * self.eHb[None, :] / self.gmw_hb
        return ua.squeeze()

    def _oxyhemoglobin_absorption(self, f_hbo2: torch.Tensor) -> torch.Tensor:
        """
        Compute the oxyhemoglobin absorption coefficient using the Manojlovic model.

        Args:
            f_hbo2: Oxyhemoglobin volume fraction

        Returns:
            Oxyhemoglobin absorption coefficient

        Raises:
            ValueError: If oxyhemoglobin volume fraction is not 1D or out of valid
        """
        if f_hbo2.ndim != 1:
            raise ValueError(f"f_hbo2 must be 1D tensor, got {f_hbo2.ndim}D")
        self.validate_tensor_range(
            f_hbo2, 0.001, 0.05, "oxyhemoglobin volume fraction f_hbo2"
        )

        # Compute absorption coefficient in m^-1
        ua = (
            math.log(10)
            * self.cHb
            * f_hbo2[:, None]
            * self.eHbO2[None, :]
            / self.gmw_hb
        )
        return ua.squeeze()

    def _bilirubin_absorption(self, f_brub: torch.Tensor) -> torch.Tensor:
        """
        Compute the bilirubin absorption coefficient using the Manojlovic model.

        Args:
            f_brub: Bilirubin concentration in mM

        Returns:
            Bilirubin absorption coefficient

        Raises:
            ValueError: If bilirubin concentration is not 1D or out of valid range
        """
        if f_brub.ndim != 1:
            raise ValueError(f"f_brub must be 1D tensor, got {f_brub.ndim}D")
        self.validate_tensor_range(
            f_brub, 10**-7, 0.1, "bilirubin concentration f_brub [mM]"
        )

        # Compute absorption coefficient in m^-1 (f_brub in mM!)
        a_brub = f_brub[:, None] * self.eBrub[None, :] / 1000
        return a_brub.squeeze()

    def _cytochrome_c_ox_absorption(self, f_cyto: torch.Tensor) -> torch.Tensor:
        """
        Compute the oxidised cytochrome c absorption coefficient
        using the Manojlovic model.

        Args:
            f_cyto: Oxidised cytochrome c concentration in mM

        Returns:
            Oxidised cytochrome c absorption coefficient

        Raises:
            ValueError: If oxidised cytochrome c concentration is not 1D
                or out of valid range
        """
        if f_cyto.ndim != 1:
            raise ValueError(f"f_cyto must be 1D tensor, got {f_cyto.ndim}D")
        self.validate_tensor_range(
            f_cyto, 10**-7, 2, "oxidised cytochrome c concentration f_cyto [mM]"
        )

        # Compute absorption coefficient in m^-1 (f_cyto in mM!)
        ua = f_cyto[:, None] * self.e_cyto_ox[None, :] / 1000
        return ua.squeeze()

    def _cytochrome_c_red_absorption(self, f_cyto: torch.Tensor) -> torch.Tensor:
        """
        Compute the reduced cytochrome c absorption coefficient
        using the Manojlovic model.

        Args:
            f_cyto: Reduced cytochrome c concentration in mM

        Returns:
            Reduced cytochrome c absorption coefficient

        Raises:
            ValueError: If reduced cytochrome c concentration is not 1D
                or out of valid range
        """
        if f_cyto.ndim != 1:
            raise ValueError(f"f_cyto must be 1D tensor, got {f_cyto.ndim}D")
        self.validate_tensor_range(
            f_cyto, 10**-7, 2, "reduced cytochrome c concentration f_cyto [mM]"
        )

        # Compute absorption coefficient in m^-1 (f_cyto in mM!)
        ua = f_cyto[:, None] * self.e_cyto_red[None, :] / 1000
        return ua.squeeze()

    def refractive_index(self) -> torch.Tensor:
        """
        Compute the refractive index using the Manojlovic model.

        Returns:
            Refractive index
        """
        n = (
            1.309
            - 4.346 * 100 * self.wavelengths**-2
            + 1.6065 * 10**9 * self.wavelengths**-4
            - 1.2811 * 10**14 * self.wavelengths**-6
        )
        return n.squeeze()

    def anisotropy(self) -> torch.Tensor:
        """
        Compute the anisotropy using the Manojlovic model.

        Returns:
            Anisotropy factor
        """
        g = 0.62 + 29 * 10 ** (-5) * self.wavelengths
        return g.squeeze()

    def mu_a_epi(self, f_m: torch.Tensor) -> torch.Tensor:
        """
        Compute the absorption coefficient of the epidermis using the Manojlovic model.

        Args:
            f_m: Melanin volume fraction

        Returns:
            Absorption coefficient of the epidermis
        """
        return self._melanin_absorption(f_m) + self._base_absorption()

    def mu_a_dermis(
        self,
        f_hb: torch.Tensor,
        f_hbo2: torch.Tensor,
        f_brub: torch.Tensor,
        f_co: torch.Tensor,
        f_coo2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the absorption coefficient of the dermis using the Manojlovic model.

        Args:
            f_hb: Hemoglobin volume fraction
            f_hbo2: Oxyhemoglobin volume fraction
            f_brub: Bilirubin millimolar concentration
            f_co: Reduced cytochrome c millimolar concentration
            f_coo2: Oxidised cytochrome c millimolar concentration

        Returns:
            Absorption coefficient of the dermis
        """
        return (
            self._hemoglobin_absorption(f_hb)
            + self._oxyhemoglobin_absorption(f_hbo2)
            + self._bilirubin_absorption(f_brub)
            + self._cytochrome_c_red_absorption(f_co)
            + self._cytochrome_c_ox_absorption(f_coo2)
            + self._base_absorption()
        )

    def _reduced_scattering(
        self, a: torch.Tensor, f_ray: torch.Tensor, b_mie: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the reduced scattering coefficient using the Manojlovic model.

        Args:
            a: Total scattering amplitude
            f_ray: Rayleigh scattering fraction
            b_mie: Mie scattering exponent

        Returns:
            Reduced scattering coefficient

        Raises:
            ValueError: If input tensors are not 1D or if f_ray is not in [0, 1]
        """
        if not all(tensor.ndim == 1 for tensor in [a, f_ray, b_mie]):
            raise ValueError("All input tensors must be 1D")
        self.validate_tensor_range(a, 2000, 8000, "scattering amplitude a [m^-1]")
        self.validate_tensor_range(
            f_ray, 10**-8, 10**-6, "Rayleigh scattering fraction f_ray"
        )

        # Compute reduced scattering coefficient in m^-1
        s = a[:, None] * (
            f_ray[:, None]
            * (self.wavelengths[None, :] * NM_TO_M / REFERENCE_WAVELENGTH_M)
            ** RAYLEIGH_EXPONENT
            + (1 - f_ray[:, None])
            * (self.wavelengths[None, :] * NM_TO_M / REFERENCE_WAVELENGTH_M)
            ** -b_mie[:, None]
        )
        return s.squeeze()

    def scattering(
        self, a: torch.Tensor, f_ray: torch.Tensor, b_mie: torch.Tensor
    ) -> torch.Tensor:
        """Compute the scattering coefficient using the Manojlovic model."""
        return self._reduced_scattering(a, f_ray, b_mie) / (
            1 - self.anisotropy()[None, :]
        )


@dataclass(frozen=True)
class LanConstants:
    # for simulations
    mu_a_range: tuple[float, float] = (0, 1000)  # m^-1
    mu_s_range: tuple[float, float] = (10000, 35000)  # m^-1
    g_range: tuple[float, float] = (0.8, 0.9999)
    n: float = (
        1.35  # not explicitly stated in the paper, rather assume index matched boundary
    )
    n_samples_sim: int = 5000
    n_photons: int = (
        10**8
    )  # not explicitly stated in the paper, use high amount (due to low sample amount)

    # for sampling from the surrogate model
    wavelengths: np.ndarray = field(
        default_factory=lambda: np.linspace(
            450, 650, 101, endpoint=True, dtype=np.float64
        )
        * 10**-9
    )
    n_samples_surrogate: int = 100000
    # for fitting/ data inference
    sao2_range: tuple[float, float] = (0.001, 1)
    vhb_range: tuple[float, float] = (0.001, 1)
    a_mie_range: tuple[float, float] = (250, 6000)  # m^-1
    b_mie_range: tuple[float, float] = (0.1, 4)


@dataclass(frozen=True)
class ManojlovicConstants:
    # for simulations
    f_mel_range: tuple[float, float] = (0.001, 0.05)
    f_hb_range: tuple[float, float] = (0.001, 0.05)
    f_hbo2_range: tuple[float, float] = (0.001, 0.05)
    f_brub_range: tuple[float, float] = (10**-7, 0.1)  # mM
    f_co_range: tuple[float, float] = (10**-7, 2)  # mM
    f_coo2_range: tuple[float, float] = (10**-7, 2)  # mM
    a_mie_range: tuple[float, float] = (2000, 8000)  # m^-1
    b_mie: float = 1.2
    f_ray: float = 10**-7
    d_epi: float = 0.0001  # m
    d_dermis: float = 0.01  # m
    n_samples_sim: int = 70000
    n_photons: int = 10**6  # not explicitly stated in the paper, use generic amount

    # for sampling from the surrogate model
    wavelengths: np.ndarray = field(
        default_factory=lambda: np.linspace(
            300, 1000, 351, endpoint=True, dtype=np.float64
        )
        * 10**-9
    )
    n_samples_surrogate: int = 100000


@dataclass(frozen=True)
class TsuiConstants:
    # for simulations
    mu_a_1_range: tuple[float, float] = (10, 500)  # m^-1
    mu_a_2_range: tuple[float, float] = (10, 500)
    mu_a_3_range: tuple[float, float] = (100, 35000)
    mu_a_4_range: tuple[float, float] = (1, 1500)
    mu_s_1_range: tuple[float, float] = (10000, 100000)  # m^-1
    mu_s_2_range: tuple[float, float] = (1000, 50000)
    mu_s_3_range: tuple[float, float] = (1000 * 1.35, 50000 * 1.35)
    mu_s_4_range: tuple[float, float] = (1000, 50000)
    f_w: float = 0.7
    g_1: float = 0.92
    g_2: float = 0.75
    g_3: float = 0.75
    g_4: float = 0.715
    n: float = 1.42
    d_1_range: tuple[float, float] = (5 * 10**-6, 30 * 10**-6)  # m
    d_2_range: tuple[float, float] = (5 * 10**-6, 30 * 10**-6)
    d_3_range: tuple[float, float] = (10 * 10**-6, 60 * 10**-6)
    n_samples_sim: int = 30000
    n_photons: int = 10**8

    # for sampling from the surrogate model
    wavelengths: np.ndarray = field(
        default_factory=lambda: np.linspace(
            460, 760, 151, endpoint=True, dtype=np.float64
        )
        * 10**-9
    )
    n_samples_surrogate: int = 100000
    # for fitting/ data inference
    a_1_range: tuple[float, float] = (10**7, 5 * 10**8)  # m^-1
    a_2_range: tuple[float, float] = (10**7, 5 * 10**8)
    a_4_range: tuple[float, float] = (10**7, 5 * 10**8)
    b_1_range: tuple[float, float] = (1, 2)
    b_2_range: tuple[float, float] = (1, 2)
    b_4_range: tuple[float, float] = (1, 2)
    f_m_range: tuple[float, float] = (0.01, 0.25)
    f_b_range: tuple[float, float] = (0, 0.005)
    sao2_range: tuple[float, float] = (0, 1)
