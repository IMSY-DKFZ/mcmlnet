"""
Absorption coefficient (mu_a) and scattering coefficient (mu_s) computation.

Copyright (c) 2016-2023 German Cancer Research Center,
Division of Intelligent Medical Systems
All rights reserved.

Redistribution and use in source and binary forms, with or
without modification, are permitted provided that the
following conditions are met:

 * Redistributions of source code must retain the above
   copyright notice, this list of conditions and the
   following disclaimer.

 * Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the
   following disclaimer in the documentation and/or other
   materials provided with the distribution.

 * Neither the name of the German Cancer Research Center,
   nor the names of its contributors may be used to endorse
   or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

"""

import math
from typing import Any

from numpy.typing import NDArray

from mcmlnet.utils.haemoglobin_extinctions import (
    get_haemoglobin_extinction_coefficients,
)


class Ua:
    """
    Helper class used to compute the absorption coefficient (Ua) used to
    generate tissue models.
    """

    def __init__(
        self,
        bvf: float = 0.04,
        c_hb: float = 150.0,
        sao2: float = 1.0,
    ):
        """
        Initialize chromophore concentrations for computation of extinction
        coefficients.

        Args:
            bvf: blood volume fraction. This refers to the percentage of
                blood relative to the volume of tissue
            c_hb: concentration of hemoglobin in blood
            sao2: oxygen concentration
            mvf: melanin volume fraction in relation to volume of tissue
            c_beta_carotene_ug_pro_dl: concentration of beta-carotene
            c_bili: concentration of bilirubin in bile. Computed as
                follows: rho_bil ~ 1374 g/L and since bile is composed
                of ~ 0.2% bilirubin: cBili ~ 0.2 * (1374 g/L) = 274.8
                g/L. Note that this is specific for bile. Another value
                has to be specified for other type of fluid.
            water: water volume fraction
            lipid: lipid volume fraction
            bili: bilirubin volume fraction
        """

        # Molecular weights
        gmw_hb = 64500  # hemoglobin

        # Blood related parameters
        self.bvf = bvf  # %
        self.cHb = c_hb  # g*Hb/L
        self.saO2 = sao2  # %
        self.gmw_hb = gmw_hb  # Gram molecular weight Hb

        # Get extinction coefficients
        (
            self.e_hbo2,
            self.e_hb,
            self.hb_wv_bounds,
        ) = get_haemoglobin_extinction_coefficients()

    def __call__(self, wavelength: float | NDArray[Any]) -> float | NDArray[Any]:
        r"""
        Determine absorption coefficient [1/m] for hemoglobin. For more details
        on this equation, please refer to: https://omlc.org/spectra/. The
        absorption coefficient is computed according to:

        $\mu_a = \log(10) \sum_{i=1}^{N} c_i \cdot \epsilon_i(\lambda)$

        where $\epsilon_i$ refers to the extinction coefficient of each molecule
        and is computed for each wavelength by interpolation with
        `scipy.interpolate.interp1d`, and $c_i$ represents the effective
        concentration of each molecule after taking into account all types of
        concentration: tissue volume fraction, volume fraction within other
        liquid mediums (e.g. blood), etc.

        Args:
            wavelength: wavelength in meters

        Returns:
            effective absorption coefficient corresponding to the
            molecular composition of the tissue
        """
        ua_haemoglobin = (
            math.log(10)
            * self.cHb
            * (
                self.saO2 * self.e_hbo2(wavelength)
                + (1 - self.saO2) * self.e_hb(wavelength)
            )
            * self.bvf
            / self.gmw_hb
        )

        return ua_haemoglobin

    def get_wavelength_bounds(self) -> dict[Any, Any]:
        """
        Extracts the wavelength boundaries for each molecule supported by this class.

        Returns:
            dictionary containing the boundaries for each molecule. Each
            element contains a tuple of `(min, max)`.
        """
        return {"hb": self.hb_wv_bounds}


class UsgJacques:
    """
    Helper class to compute the scattering coefficient (u_s) and anisotropy (g)
    of a tissue model.
    """

    def __init__(self) -> None:
        # Considering Mie scattering alone by default
        self.a_ray = 0.0 * 100.0
        self.a_mie = 20.0 * 100.0
        self.b_mie = 1.286

        self.g = 0.0

    def __call__(self, wavelength: NDArray[Any]) -> tuple[NDArray[Any], float]:
        r"""
        Calculate the scattering parameters relevant for monte carlo simulation.
        Uses equation (2) from: Optical properties of biological tissues: a Review
        (Jacques 2013).

        $$
        \mu_s = \dfrac {a_{ray} \cdot (\lambda / (500\cdot 10^{-9}))^{-4} +
        a_{mie} \cdot (\lambda / (500\cdot 10^{-9}))^{-b_{mie}}} {1 - g}
        $$

        Args:
            wavelength: wavelength of the incident light [m]

        Returns:
            scattering coefficient us [1/m]
            anisotropy factor g
        """
        norm_wavelength = wavelength / (500 * 10**-9)

        us_ray = self.a_ray * norm_wavelength ** (-4)
        us_mie = self.a_mie * norm_wavelength ** (-self.b_mie)

        us = (us_ray + us_mie) / (1 - self.g)

        return us, self.g
