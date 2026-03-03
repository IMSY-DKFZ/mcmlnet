"""
This module contains the functions to adapt the simulated reflectance spectra
to the TIVITA camera wavelengths.

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

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FILTER_RESPONSE_NAME = "filter_response"
WAVELENGTHS_NAME = "wavelengths"
OPTICAL_COMPONENTS_NAME = "optical_components"
WHITE_NAME = "white"
DARK_NAME = "dark"
NR_BANDS_NAME = "nr_bands"


@dataclass
class ImagingSystem:
    """
    Represents an imaging system that models the filtering, spectral response, and
    optical characteristics of an imaging system.

    This class is designed to handle various attributes that define
    the behavior and characteristics of an imaging system. This includes filter
    responses across spectral bands, wavelengths, and optional optical effects. It
    also accounts for white and dark reference data. The attributes of the imaging
    system can be initialized and default values are assigned where applicable.

    Attributes:
        f: Filter responses of a camera.
        wv: Wavelengths for each value in the data for each band
            of the camera.
        nb: Number of bands corresponding to the camera.
        oc: Optical components, such as quantum efficiency, transmission, and
            irradiance of a light source. Defaults to an array of ones if not
            provided.
        w: White reference data. Defaults to an array of ones if
            not provided.
        d: Dark reference data. Defaults to an array of zeros if
            not provided.
    """

    f: NDArray[Any]  # filter responses of a camera
    wv: NDArray[
        Any
    ]  # wavelengths for each value in the data for each band of the camera
    nb: int = field(
        default_factory=lambda: 0
    )  # number of bands corresponding to the camera
    oc: NDArray[Any] = field(
        default_factory=lambda: np.array([])
    )  # all optical component effects: quantum efficiency, transmission, etc.
    w: NDArray[Any] = field(default_factory=lambda: np.array([]))  # white
    d: NDArray[Any] = field(default_factory=lambda: np.array([]))  # dark

    def __post_init__(self) -> None:
        """
        Initializes default values for attributes if they are not already set.

        This method is automatically called after the object's initialization to ensure
        that the attributes `w`, `d`, and `oc` have appropriate default values, based
        on the size of the attribute `wv`.

        Raises:
            ValueError: If `wv` is not defined or has an invalid shape.
        """
        if self.nb == 0:
            self.nb = self.f.shape[0]
        if self.w.size == 0:
            self.w = np.ones_like(self.wv)
        if self.d.size == 0:
            self.d = np.zeros_like(self.wv)
        if self.oc.size == 0:
            self.oc = np.ones_like(self.wv)


def get_imaging_system(imaging_system: dict[str, Any]) -> ImagingSystem:
    """
    Extracts and constructs an ImagingSystem object from the provided dictionary of
    imaging system data.

    Args:
        imaging_system: Dictionary containing keys for filter response name,
            wavelengths, number of bands, and optionally optical components,
            white reference data, and dark reference data. The key names used
            are defined in constants. This dictionary should at least contain
            the camera filter responses, and the wavelengths.

    Returns:
        ImagingSystem: A constructed ImagingSystem object based on the provided data.
    """
    im_system = ImagingSystem(
        f=imaging_system[FILTER_RESPONSE_NAME],
        wv=imaging_system[WAVELENGTHS_NAME],
        nb=imaging_system.get(NR_BANDS_NAME, 0),  # avoid mypy error
        oc=imaging_system.get(OPTICAL_COMPONENTS_NAME, np.array([])),
        w=imaging_system.get(WHITE_NAME, np.array([])),
        d=imaging_system.get(DARK_NAME, np.array([])),
    )
    return im_system


def switch_reflectances(
    df: pd.DataFrame, new_wavelengths: NDArray[Any], new_reflectances: NDArray[Any]
) -> pd.DataFrame:
    """
    Changes the "reflectances" in a pandas DataFrame (with MultiIndex). A copy of
    `df` is created while dropping column "reflectances" from axis=1 and level=0;
    a new DataFrame is created with the new reflectances and concatenated to the
    created copy.

    Args:
        df: DataFrame with MultiIndex as columns. The column
            "reflectances" should be in axis=1 and level=0
        new_wavelengths: iterable with the new wavelengths to use as
            column names in the returned DataFrame MultiIndex
        new_reflectances: array with the new reflectances, length of
            `new_wavelengths` should be equal to
            `new_reflectances.shape[1]`

    Returns:
        DataFrame with the new "reflectances"
    """
    if len(new_wavelengths) != new_reflectances.shape[1]:
        raise ValueError(
            f"Length of new_reflectances does not match new_reflectances: "
            f"{len(new_wavelengths)}!={new_reflectances.shape[1]}"
        )
    df_dropped = df.drop("reflectances", axis=1, level=0, inplace=False).reset_index(
        drop=True
    )
    new_df = pd.DataFrame(new_reflectances, columns=new_wavelengths)
    new_df.columns = pd.MultiIndex.from_product([["reflectances"], new_df.columns])
    new_df = pd.concat([df_dropped, new_df], axis=1)
    return new_df


def spectral_reflectance_to_camera_color(
    imaging_system: dict[str, NDArray[Any] | int], r: NDArray[Any]
) -> NDArray[Any]:
    """
    Transforms simulated spectral reflectances into simulated camera intensities.

    Args:
        imaging_system: a dictionary containing the information needed
            to transform the data. The following keys are expected. Each
            value in the dictionary should be an array.
        r: array containing reflectances with dimensions (nr_samples x
            nr_bands)

    Returns:
        array containing simulated camera intensities
    """
    i = get_imaging_system(imaging_system)  # short alias for the imaging system
    # initialize array on which to save values
    camera_color = np.zeros((_nr_samples_to_transform(r), i.nb))

    # iterate over bands of the imaging system
    for k in range(i.nb):
        combined_imaging_system = i.oc * i.f[k, :] * (i.w - i.d)
        vectorized_response = combined_imaging_system * r
        # integrate over wavelengths to get camera response for band k:
        color_k = np.trapezoid(y=vectorized_response, x=i.wv)
        camera_color[:, k] = color_k
    return np.squeeze(camera_color)


def spectral_irradiance_to_camera_color(
    imaging_system: dict[str, NDArray[Any] | int], s: NDArray[Any]
) -> NDArray[Any]:
    """
    Transforms spectral irradiance into simulated camera intensities (camera color).

    Args:
        imaging_system: a dictionary containing the information needed
            to transform the data. The following keys are expected. Each
            value in the dictionary should be an array.
        s: array containing irradiance with dimensions (nr_samples x
            nr_bands)

    Returns:
        array containing simulated camera intensities
    """
    i = get_imaging_system(imaging_system)  # short alias for the imaging system
    camera_color = np.zeros((_nr_samples_to_transform(s), i.nb))

    # iterate over bands of the imaging system
    for k in range(i.nb):
        combined_imaging_system = i.oc * i.f[k, :]
        vectorized_response = combined_imaging_system * (s - i.d)
        # camera response for band k:
        color_k = np.trapezoid(vectorized_response, i.wv)
        camera_color[:, k] = color_k
    return np.squeeze(camera_color)


def camera_color_to_camera_reflectance(
    imaging_system: dict[str, NDArray[Any] | int], camera_color: NDArray[Any]
) -> NDArray[Any]:
    """
    Transforms camera intensities to camera reflectances. Notice that this is
    equivalent to normalizing each camera band (channel) by the integral over
    the entire wavelength range.

    Args:
        imaging_system: a dictionary containing the information needed
            to transform the data. The following keys are expected. Each
            value in the dictionary should be an array.
        camera_color: array containing simulated camera intensities with
            dimensions (nr_samples x nr_bands)

    Returns:
        array containing simulated camera reflectances
    """
    if len(camera_color.shape) == 1:
        camera_color = camera_color[np.newaxis, :]
    # normalize each band
    white = get_white_color(imaging_system)
    camera_reflectance: NDArray[Any] = camera_color / white[np.newaxis, ...]
    camera_reflectance = np.squeeze(camera_reflectance)
    return camera_reflectance


def get_white_color(imaging_system: dict[str, NDArray[Any] | int]) -> NDArray[Any]:
    """
    Computes "white" reference for normalization of camera intensities, depends
    only on imaging system.

    Args:
        imaging_system: a dictionary containing the information needed
            to transform the data. The following keys are expected. Each
            value in the dictionary should be an array.

    Returns:
        array containing "white" reference
    """
    i = get_imaging_system(imaging_system)  # short alias for imaging system

    white = np.zeros(i.nb)
    for k in range(i.nb):
        combined_imaging_system = i.oc * i.f[k, :] * (i.w - i.d)
        white[k] = np.trapezoid(y=combined_imaging_system, x=i.wv)
    return white


def transform_reflectance(
    imaging_system: dict[str, NDArray[Any] | int], r: NDArray[Any]
) -> NDArray[Any]:
    """
    Given a set of reflectances (nxm), transform them to what camera reflectance
    space (no noise added).

    Args:
        imaging_system: a dictionary containing the information needed
            to transform the data. The following keys are expected. Each
            value in the dictionary should be an array.
        r: set of reflectance measurements (nxm). These can e.g. be the
            output of a Monte Carlo simulation.

    Returns:
        the measurement transformed to show what the imaging system
        would measure (nxv)
    """
    camera_color = spectral_reflectance_to_camera_color(imaging_system, r)
    camera_reflectance = camera_color_to_camera_reflectance(
        imaging_system, camera_color
    )
    return camera_reflectance


def transform_color(
    imaging_system: dict[str, NDArray[Any] | int], s: NDArray[Any]
) -> NDArray[Any]:
    """
    Given a set of spectrometer irradiance measurements (nxm), transform them to
    what the camera reflectance space (no noise added). The difference to
    transform_reflectance is that S is not dark and white light corrected.

    Args:
        imaging_system: a dictionary containing the information needed
            to transform the data. The following keys are expected. Each
            value in the dictionary should be an array.
        s: set of spectrometer measurements (nxm)

    Returns:
        the measurement transformed to show what the imaging system
        would measure (nxv)
    """
    camera_color = spectral_irradiance_to_camera_color(imaging_system, s)
    camera_reflectance = camera_color_to_camera_reflectance(
        imaging_system, camera_color
    )
    return np.squeeze(camera_reflectance)


def _nr_samples_to_transform(x: NDArray[Any]) -> int:
    """
    Computes the number of samples to be transformed. Uses dimension 0 of
    array as #samples.

    Args:
        x: array containing samples, dimension 0 corresponds to #
            samples

    Returns:
        int, number of samples in the array
    """
    if len(x.shape) == 1:
        n = 1
    else:
        n = x.shape[0]
    return n


def adapt_to_camera(
    batch: pd.DataFrame,
    imaging_system: dict[Any, Any],
    reflectance_columns: str | Iterable[Any] | None = "reflectances",
    origin: str = "reflectance",
    output: str | None = None,
) -> pd.DataFrame:
    """
    Manipulation of loaded reflectance data to receive reflectances at new
    wavelengths. The values in the `reflectances` columns of `batch` are
    transformed to the camera.

    Args:
        batch: dataframe or path to folder containing MCML simulations
        imaging_system: a dictionary containing the information needed
            to transform the data. The following keys are expected. Each
            value in the dictionary should be an array.
        reflectance_columns: columns where the reflectances are stored
        origin: one of `raw` or `reflectance`. Reflectances
            are raw counts that have been white and dark normalized
        output: If str, the results are saved to a csv file indicated by
            "output"

    Returns:
        reflectance data at new wavelengths
    """
    df = batch.copy(deep=True)
    # get simulated camera intensities
    if reflectance_columns is not None:
        r = df[reflectance_columns]
    else:
        r = df.values
    if origin == "reflectance":
        Cc = transform_reflectance(imaging_system, r)
    elif origin == "raw":
        Cc = transform_color(imaging_system, r)
    else:
        raise ValueError(f"Could not interpret type: {origin}")
    if len(Cc.shape) == 1:
        Cc = Cc[np.newaxis, ...]
    bands_camera = np.arange(Cc.shape[1])

    df = switch_reflectances(df, bands_camera, Cc)
    if "penetration" in df.keys():
        p_adapted = transform_reflectance(imaging_system, df.penetration)
        df.drop("penetration", axis=1, level=0, inplace=True)
        penetration_dict = {
            ("penetration", nw): p_adapted[:, i] for i, nw in enumerate(bands_camera)
        }
        penetration_df = pd.DataFrame(penetration_dict)
        df = pd.concat([df, penetration_df], axis=1)
        if np.any(df.isna()):
            raise ValueError(
                "Found NaN values in data frame after adaptation to camera space"
            )
        assert np.all(df.penetration.values == p_adapted)

    if output is not None:
        df.to_csv(output, index=False)
    return df
