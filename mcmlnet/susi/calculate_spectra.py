"""
Top-level module for calculating spectra using GPUMCML.

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

import os
import warnings
from subprocess import TimeoutExpired
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from rich.progress import track

from mcmlnet.constants import MCML_PATH, MCO_PATH
from mcmlnet.susi.sim import GenericMcFactory, MciWrapper, SimWrapper

# Validate paths if they exist
if MCML_PATH and not os.path.exists(MCML_PATH):
    warnings.warn(f"MCML_PATH does not exist: {MCML_PATH}", stacklevel=2)
if MCO_PATH and not os.path.exists(MCO_PATH):
    warnings.warn(f"MCO_PATH does not exist: {MCO_PATH}", stacklevel=2)


def get_mcml_executable() -> str:
    """Get the MCML executable path with validation.

    Returns:
        str: Path to MCML executable
    """
    if not MCML_PATH:
        raise RuntimeError("MCML_PATH environment variable is not set")
    if not os.path.exists(MCML_PATH):
        raise RuntimeError(f"MCML executable not found at: {MCML_PATH}")
    return MCML_PATH


def get_mco_path() -> str:
    """Get the MCO file path with validation.

    Returns:
        str: Path to MCO file
    """
    if not MCO_PATH:
        raise RuntimeError("MCO_PATH environment variable is not set")
    if not os.path.exists(MCO_PATH):
        raise RuntimeError(f"MCO file not found at: {MCO_PATH}")
    return MCO_PATH


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


def insert_opt_params(
    df: pd.DataFrame,
    ua: dict[str, NDArray[Any]],
    us: dict[str, NDArray[Any]],
    wavelengths: NDArray[Any],
    layers: list[str],
) -> pd.DataFrame:
    """
    Inserts absorption and scattering coefficients (ua and us) into dataframe as
    columns named `ua_layerN` where `N` indicates the layer number. These parameters
    are stored for each wavelength.

    Args:
        df: dataframe where optical parameters are inserted
        ua: absorption coefficient
        us: scattering coefficient
        wavelengths: array of wavelengths in meters
        layers: list of layer names in data frame

    Returns:
        new pandas data frame with new columns appended
    """
    new_cols = {}
    for layer in layers:
        for i, nw in enumerate(wavelengths):
            new_cols[(f"ua_{layer}", nw)] = ua[layer][:, i]
            new_cols[(f"us_{layer}", nw)] = us[layer][:, i]
    new_df = pd.DataFrame(new_cols)
    new_df = pd.concat([df, new_df], ignore_index=False, axis=1)
    return new_df


def calculate_spectrum_for_batch(
    batch: pd.DataFrame,
    wavelengths: NDArray[Any],
    nr_photons: int,
    mci_base_folder: str | None = None,
    batch_id: str = "",
    ignore_a: bool = False,
    mco_file: str = "",
    verbose: bool = True,
    timeout: float | None = None,
) -> pd.DataFrame:
    """
    Take a single batch without simulations and generate the simulations with MCML.

    Args:
        batch: dataframe containing the physiological and optical
            properties for each layer
        wavelengths: the wavelengths to simulate in international units
        nr_photons: the number of photons
        mci_base_folder: the folder in which the mci inputs for the
            simulations are saved
        batch_id: ID to be used to identify this batch during
            simulations
        ignore_a: ignore absorption detection during simulation. Will
            speed up simulations a little but will prevent computation
            of the penetration depth.
        mco_file: the path to a file where MCML results will be stored.
        verbose: show output from MCML subprocess
        timeout: time to wait until killing MCML subprocess

    Returns:
        data frame with simulated reflectances
    """
    # Get validated MCML executable path
    mcml_executable = get_mcml_executable()

    if mci_base_folder is None:
        mci_base_folder = os.path.dirname(mcml_executable)
    simulated_data_folder = mcml_executable
    layers = list(batch.columns.levels[0])
    nr_instances = batch.shape[0]
    factory = GenericMcFactory(len(layers))

    # create folder for mci files if not exists
    mci_folder = os.path.join(mci_base_folder, "mci")
    if not os.path.exists(mci_folder):
        os.makedirs(mci_folder)

    # add reflectance columns to dataframe
    column_head = [
        np.array(["reflectances"] * len(wavelengths)),
        wavelengths,
    ]

    reflectance_df = pd.DataFrame(np.nan, index=batch.index, columns=column_head)
    batch = pd.concat([batch, reflectance_df], axis=1)

    # Setup simulation wrapper
    sim_wrapper = SimWrapper(ignore_a=ignore_a)
    sim_wrapper.set_mcml_executable(mcml_executable)
    mci_filename = os.path.join(mci_folder, "Bat_" + "NA_" + str(batch_id) + ".mci")
    sim_wrapper.set_mci_filename(mci_filename)

    # Setup a tissue instance. A tissue instance is one "row" in the batch,
    # holding the physiological parameters of all the layers.
    tissue_instance = factory.create_tissue_instance()
    tissue_instance.set_mci_filename(sim_wrapper.mci_filename)
    tissue_instance.set_nr_photons(nr_photons)
    tissue_instance._mci_wrapper.set_nr_runs(nr_instances * wavelengths.shape[0])
    tissue_instance._mci_wrapper.create_mci_file()

    # Generate MCI file which contains list of all simulations in a Batch
    ua = {l: np.zeros((nr_instances, len(wavelengths))) for l in layers}  # noqa: E741
    us = {l: np.zeros((nr_instances, len(wavelengths))) for l in layers}  # noqa: E741

    # Setup progress bar for MCI file creation
    if verbose:
        iterator = track(range(batch.shape[0]), description="Creating MCI file")
    else:
        iterator = range(batch.shape[0])

    for i in iterator:
        # set the desired element in the dataframe to be simulated
        base_mco_filename = _get_mco_filename(
            simulated_data_folder, "NA_" + str(batch_id), i
        )
        tissue_instance.set_base_mco_filename(base_mco_filename)
        tissue_instance.set_tissue_instance(batch.loc[i, :])
        tissue_instance.update_mci_content(wavelengths)
        params = tissue_instance.get_ua_us(wavelengths=wavelengths)
        for j, layer in enumerate(layers):
            ua[layer][i] = params["ua"][j]
            us[layer][i] = params["us"][j]

    tissue_instance.write_mci_content()

    # Run simulations for computing reflectance from parameters
    sim_wrapper.run_simulation(mco_file=mco_file, verbose=verbose, timeout=timeout)

    # get information from created mco file
    if mco_file:
        mco_filename = mco_file
    else:
        mco_filename = get_mco_path()
    df = pd.read_csv(mco_filename, index_col=0, sep=",")

    reflectance_array = np.zeros((nr_instances, len(wavelengths)))
    penetration_array = np.zeros((nr_instances, len(wavelengths)))

    for i in range(nr_instances):
        for j, wavelength in enumerate(wavelengths):
            # for simulation get which mco file was created
            base_mco_filename = _get_mco_filename(
                simulated_data_folder, "NA_" + str(batch_id), i
            )
            id_mco_filename = base_mco_filename + format(wavelength, "1.2e") + ".mco"
            # get diffuse reflectance from simulation
            reflectance_array[i, j] = df.loc[id_mco_filename]["Diffuse"]
            if "Penetration" in df.keys():
                penetration_array[i, j] = df.loc[id_mco_filename]["Penetration"]

    # put the reflectances in the batch:
    batch = switch_reflectances(batch, wavelengths, reflectance_array)

    # insert optical properties into batch
    batch = insert_opt_params(
        df=batch, ua=ua, us=us, wavelengths=wavelengths, layers=layers
    )

    # add penetration depth only if ignore_a is set to False
    if "Penetration" in df.keys() and not ignore_a:
        column_head = [
            np.array(["penetration"] * len(wavelengths)),
            wavelengths,
        ]

        penetration_df = pd.DataFrame(
            penetration_array / 100, index=batch.index, columns=column_head
        )
        batch = pd.concat([batch, penetration_df], axis=1)

    # delete created mco file
    os.remove(mco_filename)
    os.remove(mci_filename)
    return batch


def _get_mco_filename(simulation_folder: str, batch_id: str, simulation_id: int) -> str:
    """
    Creates the filename that will be created by MCML and that can be used when
    reading the dataframe from the MCML output.

    Args:
        simulation_folder: path to folder where simulations are stored
        batch_id: string identifying the batch that is being simulated
        simulation_id: integer identifying the simulations that is to be
            performed. Usually this is the number of the row in the
            batch data frame

    Returns:
        string identifying the .mco file that will be created by MCML
    """
    prefix = os.path.split(simulation_folder)[1]
    return str(prefix) + "_Bat_" + batch_id + "_Sim_" + str(simulation_id) + "_"


def check_simulation_batch(
    batch: pd.DataFrame,
    wavelengths: NDArray[Any],
    n_layers: int,
) -> None:
    """
    Sanity check the shape and data ranges of a structured MultiIndex
    physical parameter dataframe.

    Args:
        batch: dataframe containing structured physical optical parameters
        wavelengths: the wavelengths to simulate in international units
        n_layers: number of layers in tissue model

    Returns:
        None
    """
    # n-layered tissue model, with 5 physic. param.s per layer
    n_physical_params_per_layer = 5
    expected_tissue_properties = ["ua", "us", "g", "n", "d"]

    # assert correct multiheader shape
    if batch.columns.nlevels != 3:
        raise ValueError("Batch dataframe must have a multiindex with 3 levels!")
    if batch.columns.levels[0].shape[0] != n_layers:
        raise ValueError("Number of layers in dataframe does not match n_layers!")
    if batch.columns.levels[1].shape[0] != len(wavelengths):
        raise ValueError(
            "Number of wavelengths in dataframe does not match n_wavelengths!"
        )
    if batch.columns.levels[2].shape[0] != n_physical_params_per_layer:
        raise ValueError(
            "Number of physical parameters in dataframe "
            "does not match 5 physical params per layer!"
        )

    # check header for completeness
    for level_0 in batch.columns.levels[0]:
        # check every layer for wavelength completeness
        _wvls = batch.loc[:, pd.IndexSlice[level_0, :, :]].columns.levels[1].tolist()
        if not np.allclose(_wvls, wavelengths, rtol=0.001):
            raise ValueError(
                f"Wavelength mismatch: {_wvls} do not match {wavelengths}!"
            )
        for level_1 in batch.columns.levels[1]:
            # check all wavelengths of every layer for physical parameter completeness
            _params = batch.loc[:, pd.IndexSlice[level_0, level_1, :]].columns
            # extract the "real" parameters from the multiindex
            _params = [param[-1] for param in _params]
            if not set(expected_tissue_properties) == set(_params):
                raise ValueError(
                    f"Tissue property mismatch: {_params} do not match "
                    f"{expected_tissue_properties}!"
                )

    # extract the parameters from the dataframe and check data range
    idx = pd.IndexSlice
    if (batch.loc[:, idx[:, :, "ua"]].to_numpy() < 0).any():
        raise ValueError("Negative absorption coefficient!")
    if (batch.loc[:, idx[:, :, "us"]].to_numpy() < 0).any():
        raise ValueError("Negative scattering coefficient!")
    if (batch.loc[:, idx[:, :, "g"]].to_numpy() < -1).any():
        raise ValueError("Anisotropy <-1!")
    if (batch.loc[:, idx[:, :, "g"]].to_numpy() > 1).any():
        raise ValueError("Anisotropy >1!")
    if (batch.loc[:, idx[:, :, "d"]].to_numpy() <= 0).any():
        raise ValueError("Thickness <= 0!")
    if (batch.loc[:, idx[:, :, "n"]].to_numpy() <= 0).any():
        raise ValueError("Refractive index <= 0!")
    if (batch.loc[:, idx[:, :, "n"]].to_numpy() < 1).any():
        warnings.warn("Refractive index < 1!", stacklevel=2)


def calculate_spectrum_for_physical_batch(
    batch: pd.DataFrame,
    wavelengths: NDArray[Any],
    nr_photons: int,
    mci_base_folder: str | None = None,
    batch_id: str = "",
    ignore_a: bool = False,
    mco_file: str = "",
    verbose: bool = True,
    timeout: float | None = None,
) -> pd.DataFrame:
    """
    Take a batch or single set of physical parameter and evaluate them using MCML
    by directly addressing set_layer.

    Args:
        batch: dataframe containing physical optical parameters structured
            by layer (level 0 header), wavelength (level 1 header) and
            parameter name (level 2 header). Parameters:
            mu_a: absorption coefficient [m^-1]
            mu_s: scattering coefficient [m^-1] - NOT mu_s'!
            g: anisotropy [-]
            n: refractive index [-]
            d: thickness [m]
        wavelengths: the wavelengths to simulate in international units
        nr_photons: the number of photons
        mci_base_folder: the folder in which the mci inputs for the
            simulations are saved
        batch_id: ID to be used to identify this batch during
            simulations
        ignore_a: ignore absorption detection during simulation. Will
            speed up simulations a little but will prevent computation
            of the penetration depth.
        mco_file: the path to a file where MCML results will be stored
        verbose: show output from MCML subprocess
        timeout: time to wait until killing MCML subprocess

    Returns:
        dataframe containing the reflectance (and penetration depth (optional))
            of shape (batch size, n_wavelengths (*2))
    """
    # sanity check the physical parameter dataframe
    n_layers = len(list(set(batch.columns.levels[0])))
    check_simulation_batch(batch, wavelengths, n_layers)

    # extract the parameters from the dataframe
    nr_instances = len(batch)
    idx = pd.IndexSlice
    ua = (
        batch.loc[:, idx[:, :, "ua"]]
        .to_numpy()
        .reshape(
            nr_instances,
            n_layers,
            len(wavelengths),
        )
    )
    us = (
        batch.loc[:, idx[:, :, "us"]]
        .to_numpy()
        .reshape(
            nr_instances,
            n_layers,
            len(wavelengths),
        )
    )
    g = (
        batch.loc[:, idx[:, :, "g"]]
        .to_numpy()
        .reshape(
            nr_instances,
            n_layers,
            len(wavelengths),
        )
    )
    n = (
        batch.loc[:, idx[:, :, "n"]]
        .to_numpy()
        .reshape(
            nr_instances,
            n_layers,
            len(wavelengths),
        )
    )
    d = (
        batch.loc[:, idx[:, :, "d"]]
        .to_numpy()
        .reshape(
            nr_instances,
            n_layers,
            len(wavelengths),
        )
    )

    # Get validated MCML executable path
    mcml_executable = get_mcml_executable()

    # set mci filename
    if mci_base_folder is None:
        mci_base_folder = os.path.dirname(mcml_executable)
    simulated_data_folder = mcml_executable

    # create folder for mci files if not exists
    mci_folder = os.path.join(mci_base_folder, "mci")
    if not os.path.exists(mci_folder):
        os.makedirs(mci_folder)

    # Setup simulation wrapper
    sim_wrapper = SimWrapper(ignore_a=ignore_a)
    sim_wrapper.set_mcml_executable(simulated_data_folder)
    mci_filename = os.path.join(mci_folder, "Bat_" + "NA_" + str(batch_id) + ".mci")
    sim_wrapper.set_mci_filename(mci_filename)

    # Setup a tissue instance. A tissue instance is one "row" in the batch,
    # holding the physiological parameters of all the layers.
    tissue_instance = MciWrapper()
    # initialize empty layers
    tissue_instance.layers = [[0]] * n_layers
    tissue_instance.set_mci_filename(mci_filename)
    tissue_instance.set_nr_photons(nr_photons)
    tissue_instance.set_nr_runs(len(wavelengths) * nr_instances)
    tissue_instance.create_mci_file()

    # Generate MCI file which contains list of all simulations in a Batch
    if verbose:
        iterator = track(range(nr_instances), description="Creating MCI file")
    else:
        iterator = range(nr_instances)

    for i in iterator:
        # set the desired element in the dataframe to be simulated
        base_mco_filename = _get_mco_filename(
            simulated_data_folder, "NA_" + str(batch_id), i
        )
        tissue_instance.set_base_mco_filename(base_mco_filename)

        for idx, wavelength in enumerate(wavelengths):
            for j in range(n_layers):
                # NOTE: PAY ATTENTION to mus vs. mus' and the units of mua and mus!
                tissue_instance.set_layer(
                    j,
                    n=n[i, j, idx],
                    ua=ua[i, j, idx],
                    us=us[i, j, idx],
                    g=g[i, j, idx],
                    d=d[i, j, idx],
                )
            tissue_instance.update_mci_content(wavelength)

    tissue_instance.write_mci_content()

    # Run simulations for computing reflectance from parameters
    try:
        sim_wrapper.run_simulation(mco_file=mco_file, verbose=verbose, timeout=timeout)
    except TimeoutExpired as err:
        # delete created simulation files
        os.remove(mci_filename)
        raise TimeoutError("Simulation timed out!") from err

    # get information from created mco file
    if mco_file:
        mco_filename = mco_file
    else:
        mco_filename = get_mco_path()
    df = pd.read_csv(mco_filename, index_col=0, sep=",")

    # storage for reflectance and penetration depth
    reflectance_array = np.zeros((nr_instances, len(wavelengths)))
    penetration_array = np.zeros((nr_instances, len(wavelengths)))

    for i in range(nr_instances):
        for j, wavelength in enumerate(wavelengths):
            # for simulation get which mco file was created
            base_mco_filename = _get_mco_filename(
                simulated_data_folder, "NA_" + str(batch_id), i
            )
            id_mco_filename = base_mco_filename + format(wavelength, "1.2e") + ".mco"
            # get diffuse reflectance from simulation
            reflectance_array[i, j] = df.loc[id_mco_filename]["Diffuse"]
            if "Penetration" in df.keys():
                penetration_array[i, j] = df.loc[id_mco_filename]["Penetration"]

    # return reflectances as DataFrame
    column_head = [
        np.array(["reflectances"] * len(wavelengths)),
        wavelengths,
    ]
    reflectance_df = pd.DataFrame(np.nan, index=batch.index, columns=column_head)
    reflectance_df = switch_reflectances(reflectance_df, wavelengths, reflectance_array)

    # add penetration depth only if ignore_a is set to False
    if "Penetration" in df.keys() and not ignore_a:
        column_head = [
            np.array(["penetration"] * len(wavelengths)),
            wavelengths,
        ]
        penetration_df = pd.DataFrame(
            penetration_array / 100, index=batch.index, columns=column_head
        )
        reflectance_df = pd.concat([reflectance_df, penetration_df], axis=1)

    # delete created mco file
    os.remove(mco_filename)
    os.remove(mci_filename)
    return reflectance_df
