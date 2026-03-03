"""
This module contains the functions to adapt the simulated reflectance spectra
to the TIVITA camera wavelengths.

"""

import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from mcmlnet.susi.adapt_to_camera import adapt_to_camera
from mcmlnet.utils.logging import setup_logging

load_dotenv()
logger = setup_logging(level="info", logger_name=__name__)


def adapt_simulations_to_tivita(
    simulations: np.ndarray,
    wavelengths: np.ndarray,
    filter_type: str = "uniform",
    cache_file_name: str = "",
    cache_dir: str = os.environ["cache_dir"],
) -> np.ndarray:
    """
    Adapts the simulated reflectance spectra to the TIVITA camera wavelengths and
    caches the results if desired.

    Args:
        simulations: Containing ONLY the simulated reflectance spectra (no param.s!)
        wavelengths: Array of simulation wavelengths [m]
        filter_type: Type of filter to use for adaptation
        cache_file_name: Name of cache file for saving adapted simulations
        cache_dir: Directory to save the cache file

    Returns:
        Adapted simulations (shape: (n_simulations, n_wavelengths))
    """
    # load camera filter response and irradiance data
    filters = pd.read_csv(
        os.path.join(
            os.environ["data_dir"],
            "raw",
            "optical_components",
            f"artificial_tivita_camera_{filter_type}.csv",
        )
    )
    # load measured irradiance data
    irradiance = pd.read_csv(
        os.path.join(
            os.environ["data_dir"],
            "raw",
            "optical_components",
            "tivita_relative_irradiance_2019_04_05.txt",
        ),
        sep=" ",
        skiprows=14,
    )
    irradiance = irradiance.rename(
        columns={">>>>>Begin": "wavelength", "Spectral": "value"}
    )
    irradiance = irradiance.drop(columns="Data<<<<<")
    irradiance["wavelength"] = irradiance["wavelength"] * 1e-9

    # select & interpolate filter response and irradiance to simulation wavelengths
    wavelengths = np.round(wavelengths, decimals=12)  # avoid artifacts
    fil = filters[[c for c in filters.columns if float(c) in wavelengths]]
    irr = np.interp(
        wavelengths,
        irradiance["wavelength"].to_numpy(),
        irradiance["value"].to_numpy(),
    )

    # define imaging system
    imaging_system = {
        "filter_response": fil.to_numpy(),
        "optical_components": irr,
        "nr_bands": 100,
        "wavelengths": wavelengths,
    }

    # convert to TIVITA camera wavelengths and print some results
    multi_index = pd.MultiIndex.from_arrays(
        [["reflectances"] * len(wavelengths), wavelengths], names=["Type", "Wavelength"]
    )
    batch = pd.DataFrame(simulations, columns=multi_index)
    try:
        sims_tivita = np.load(os.path.join(cache_dir, cache_file_name))
    except OSError:
        sims_tivita = adapt_to_camera(
            batch,
            imaging_system=imaging_system,
            origin="reflectance",
        ).dropna(axis=1, inplace=False)
        sims_tivita = sims_tivita.to_numpy()
        if cache_file_name:
            logger.info(f"Saving adapted simulations to {cache_file_name}")
            np.save(os.path.join(cache_dir, cache_file_name), sims_tivita)

    return sims_tivita
