"""
Simulation and mci file handling for MCML simulations.

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
import shutil
import subprocess
from collections.abc import MutableSequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mcmlnet.constants import MCO_PATH
from mcmlnet.susi.usuag import Ua, UsgJacques
from mcmlnet.utils.logging import setup_logging

logger = setup_logging(level="info", logger_name=__name__)


def get_diffuse_reflectance(mco_filename: str) -> float:
    """Extract reflectance from mco file.

    Attention: mco_filename specifies full path.
    """
    df = pd.read_csv(MCO_PATH, index_col=0, sep=",")
    return df.loc[mco_filename]["Diffuse"]  # type: ignore [no-any-return]


def get_specular_reflectance(mco_filename: str) -> float:
    """Extract reflectance from mco file.

    Attention: mco_filename specifies full path.
    """
    df = pd.read_csv(MCO_PATH, index_col=0)
    return df.loc[mco_filename]["Specular"]  # type: ignore [no-any-return]


def get_total_reflectance(mco_filename: str) -> float:
    """Extract reflectance from mco file.

    Attention: mco_filename specifies full path.
    """
    return get_diffuse_reflectance(mco_filename) + get_specular_reflectance(
        mco_filename
    )


class AbstractTissue:
    """Initializes a abstract tissue model"."""

    def set_nr_photons(self, nr_photons: int) -> None:
        self._mci_wrapper.set_nr_photons(nr_photons)

    def set_mci_filename(self, mci_filename: str) -> None:
        self._mci_wrapper.set_mci_filename(mci_filename)

    def set_base_mco_filename(self, base_filename: str) -> None:
        self._mci_wrapper.set_base_mco_filename(base_filename)

    def get_mco_filename(self) -> str | None:
        return self._mci_wrapper.get_base_mco_filename()

    def set_wavelength(self, wavelength: float) -> None:
        self.wavelength = wavelength

    def create_mci_file(self) -> None:
        self._mci_wrapper.create_mci_file()

    def write_mci_content(self) -> None:
        self._mci_wrapper.write_mci_content()

    def update_mci_content(self, wavelengths: NDArray[Any]) -> None:
        # set layers
        uas: list[NDArray[Any]] = [
            self.uas[i](wavelengths) for i, _ in enumerate(self.uas)
        ]
        for j, wavelength in enumerate(wavelengths):
            for i, _ua in enumerate(self.uas):
                self._mci_wrapper.set_layer(
                    i,  # layer nr
                    self.ns[i],  # refraction index
                    uas[i][j],  # ua
                    self.usgs[i](wavelength)[0],  # us
                    self.usgs[i](wavelength)[1],  # g
                    self.ds[i],  # d
                )
            # now that the layers have been updated: create file
            self._mci_wrapper.update_mci_content(wavelength)

    def get_ua_us(self, wavelengths: NDArray[Any]) -> dict[str, Any]:
        """
        Queries the absorption and scattering coefficients
        for the specified wavelengths.

        Args:
            wavelengths: wavelengths in meters to query

        Returns:
            dictionary with keywords `ua` and `us` corresponding to the
            absorption and scattering coefficients respectively
        """
        ua = [self.uas[i](wavelengths) for i, _ in enumerate(self.uas)]
        us = [self.usgs[i](wavelengths)[0] for i, _ in enumerate(self.usgs)]
        return {"ua": ua, "us": us}

    def __str__(self) -> str:
        """Overwrite this method!

        print the current model
        """
        model_string = ""
        return model_string

    def __init__(
        self,
        ns: MutableSequence[float],
        uas: list[Ua],
        usgs: list[UsgJacques],
        ds: MutableSequence[float],
    ):
        self._mci_wrapper = MciWrapper()

        self.wavelength = 500.0 * 10**9  # standard wavelength, should be set.
        self.uas = uas
        self.usgs = usgs
        self.ds = ds
        self.ns = ns
        # initially create layers. these will be overwritten as soon
        # as create_mci_file is called.
        for _ in enumerate(uas):
            self._mci_wrapper.add_layer()


class AbstractMcFactory:
    """Monte Carlo Factory.

    Will create fitting models and batches, dependent on your task
    """

    def __init__(self) -> None:
        """Constructor."""

    def create_tissue_instance(self) -> AbstractTissue:
        """Create tissue instance."""
        return AbstractTissue(uas=[], usgs=[], ds=[], ns=[])


class GenericTissue(AbstractTissue):
    """Initializes a 3-layer generic tissue model."""

    def set_tissue_instance(self, df_row: pd.Series) -> None:
        """
        Take one example (one row) of a created batch and set the tissue
        to resemble the structure specified by this row.

        Args:
            df_row: one row of a data frame created by a batch.
        """
        layers = [l for l in df_row.index.levels[0] if "layer" in l]  # noqa: E741

        for i, layer in enumerate(layers):
            self.set_layer(i, df_row[layer,])

    def set_layer(self, layer_nr: int, df_layer: pd.Series) -> None:
        """Helper function to set one layer."""

        bvf = 0.0
        saO2 = 0.0
        chb = 150.0
        if layer_nr != 0:
            saO2 = self.uas[0].saO2
        a_mie = 10.0 * 100
        a_ray = 0.0
        d = 500.0 * 10**-6
        b_mie = 1.286
        n = 1.36
        g = 0.0

        if "a_ray" in df_layer:
            a_ray = np.clip(df_layer["a_ray"], 0.0, np.inf)
        if "a_mie" in df_layer:
            a_mie = np.clip(df_layer["a_mie"], 0.0, np.inf)
        if "b_mie" in df_layer:
            b_mie = np.clip(df_layer["b_mie"], 0.0, np.inf)
        if "vhb" in df_layer:
            bvf = np.clip(df_layer["vhb"], 0.0, np.inf)
        if "sao2" in df_layer:
            saO2 = np.clip(df_layer["sao2"], 0.0, np.inf)
        if "d" in df_layer:
            d = np.clip(df_layer["d"], 0.0, np.inf)
        if "n" in df_layer:
            n = np.clip(df_layer["n"], 0.0, np.inf)
        if "g" in df_layer:
            g = np.clip(df_layer["g"], 0.0, 0.9999)
        if "chb" in df_layer:
            chb = np.clip(df_layer["chb"], 0.0, np.inf)

        # Build object for absorption coefficient determination
        self.uas[layer_nr].bvf = bvf
        self.uas[layer_nr].saO2 = saO2
        self.uas[layer_nr].cHb = chb

        # and one for scattering coefficient
        self.usgs[layer_nr].a_mie = a_mie
        self.usgs[layer_nr].a_ray = a_ray
        self.usgs[layer_nr].b_mie = b_mie
        self.usgs[layer_nr].g = g
        self.ds[layer_nr] = d
        self.ns[layer_nr] = n

    def __str__(self) -> str:
        """Print the current model."""
        model_string = ""
        for i, _ua in enumerate(self.uas):
            layer_string = (
                f"layer {i}    - vhb: {self.uas[i].bvf * 100.0:.1f}%"
                f"; sao2: {self.uas[i].saO2 * 100.0:.1f}%"
                f"; a_mie: {self.usgs[i].a_mie / 100.0:.2f}cm^-1"
                f"; cHb: {self.uas[i].cHb / 100.0:.2f}g/l"
                f"; a_ray: {self.usgs[i].a_ray / 100.0:.2f}cm^-1"
                f"; b_mie: {self.usgs[i].b_mie:.3f}"
                f"; d: {self.ds[i] * 10**6:.0f}um"
                f"; n: {self.ns[i]:.2f}"
                f"; g: {self.usgs[i].g:.2f}\n"
            )
            model_string += layer_string
        return model_string

    def __init__(self, nr_layers: int) -> None:
        uas = []
        usgs = []
        for _i in range(nr_layers):
            # Use the default values of the tissue parameters
            # to create an instance of Ua along with
            # the extinction coefficients LUT
            # Also the Jacques2013 equation (1) or (2) is used to
            # compute 'us' and 'g'
            uas.append(Ua())
            usgs.append(UsgJacques())

        # Initialize thickness of each layer
        ds = np.ones(nr_layers, dtype=float) * 500.0 * 10**-6
        # Initialize refractive indices for each layer
        ns = np.ones(nr_layers, dtype=float) * 1.38
        super().__init__(ns, uas, usgs, ds)


class GenericMcFactory(AbstractMcFactory):
    def __init__(self, nr_layers: int) -> None:
        """Init function."""
        super().__init__()
        self.nr_layers = nr_layers

    def create_tissue_instance(self) -> GenericTissue:
        """Create tissue instance."""
        return GenericTissue(self.nr_layers)


class MciWrapper:
    """This class provides a wrapper to the mcml monte carlo file.

    Its purpose is to create a .mci file which the mcml simulation can use to perform
    the simulation
    """

    def __init__(self) -> None:
        self.file_version = 1.0
        self.nr_photons = 10**6
        self.nr_runs = 1
        self.dz = 0.002
        self.dr = 2.0
        self.nr_dz = 500
        self.nr_dr = 1
        self.nr_da = 1
        self.n_above = 1.0
        self.n_below = 1.0
        # initialize to 0 layers
        self.layers: list[list[float]] = []
        self._mci_content: list[str] = []

    def set_mci_filename(self, mci_filename: str) -> None:
        """Set mci filename."""
        self.mci_filename = mci_filename

    def set_base_mco_filename(self, base_filename: str) -> None:
        """Set base mco filename."""
        self.base_mco_filename = base_filename

    def get_base_mco_filename(self) -> str | None:
        """Get base mco filename."""
        return self.base_mco_filename

    def set_nr_photons(self, nr_photons: int) -> None:
        """Set nr photons."""
        self.nr_photons = nr_photons

    def add_layer(
        self,
        n: float | None = None,
        ua: float | None = None,
        us: float | None = None,
        g: float | None = None,
        d: float | None = None,
    ) -> None:
        """Adds a layer below the currently existing ones.

        Args:
            n: Refraction index of medium
            ua: absorption coefficient [1/m]
            us: scattering coefficient [1/m]
            g: anisotropy factor
            d: thickness of layer [m]
        """
        if n is None:
            n = 1.0
        if ua is None:
            ua = 0.0
        if us is None:
            us = 0.0
        if g is None:
            g = 1.0
        if d is None:
            d = 500.0 * 10**-6
        self.layers.append([n, ua, us, g, d])

    def set_layer(
        self, layer_nr: int, n: float, ua: float, us: float, g: float, d: float
    ) -> None:
        """Set a layer with a specific layer_nr (starting with layer_nr 0).

        Note that the layer must already exist, otherwise an error will occure
        """
        self.layers[layer_nr] = [n, ua, us, g, d]

    def set_file_version(self, file_version: float) -> None:
        """Set file version."""
        self.file_version = file_version

    def set_nr_runs(self, nr_runs: int) -> None:
        """Set nr runs."""
        self.nr_runs = nr_runs

    def set_dz_dr(self, dz: float, dr: float) -> None:
        """Set dz dr."""
        self.dz = dz
        self.dr = dr

    def set_nr_dz_dr_da(self, nr_dz: int, nr_dr: int, nr_da: int) -> None:
        """Set nr dz dr da."""
        self.nr_dz = nr_dz
        self.nr_dr = nr_dr
        self.nr_da = nr_da

    def set_n_medium_above(self, n_above: float) -> None:
        """Set n medium above."""
        self.n_above = n_above

    def set_n_medium_below(self, n_below: float) -> None:
        """Set n medium below."""
        self.n_below = n_below

    def create_mci_file(self) -> None:
        """Create mci file."""
        # write header
        f = open(self.mci_filename, "w")
        f.write(str(self.file_version) + " # file version\n")
        f.write(str(self.nr_runs) + " # number of runs\n\n")
        f.close()

    def write_mci_content(self) -> None:
        """Write mci content."""
        if self._mci_content:
            content = "".join(self._mci_content)
            with open(self.mci_filename, "a") as handle:
                handle.write(content)
        else:
            logger.warning("No content to write to .mci file, nothing is done.")

    def update_mci_content(self, wavelength: float) -> None:
        """This method creates the mci file at the location self.mci_filename, using 8
        decimal places of precision for the physical parameters."""

        # Generate a new mco fileName
        local_mco_filename = self.base_mco_filename + f"{wavelength:1.2e}" + ".mco"
        # write the data for run
        layer_content = (
            f"{local_mco_filename} A # output filename, ASCII/Binary\n"
            f"{self.nr_photons} # No. of photons\n"
            f"{self.dz} {self.dr} # dz, dr\n"
            f"{self.nr_dz} {self.nr_dr} {self.nr_da} # No. of dz, dr & da.\n\n"
            f"{len(self.layers)} # No. of layers\n"
            f"# n mua mus g d # One line for each layer\n"
            f"{self.n_above} # n for medium above.\n"
        )
        for layer in self.layers:
            # factors (/100.; *100.) to convert to mcml expected units:
            # Add the following, in that order: n, ua, us, g , d[cm]
            # (8 decimal places for suffficient precision (~float32)
            layer_content += (
                f"{layer[0]:.8f} {layer[1] / 100:.8f} "
                f"{layer[2] / 100:.8f} {layer[3]:.8f} "
                f"{layer[4] * 100:.8f}\n"
            )
        layer_content += f"{self.n_below} # n for medium below.\n"
        self._mci_content.append(layer_content)


class SimWrapper:
    def __init__(self, ignore_a: bool = False) -> None:
        """Init function."""
        if ignore_a:
            logger.warning(
                f"ignore_A has been set to {ignore_a}, "
                "absorption detection will be ignored, "
                "this will speed up simulations but "
                "will not allow to compute penetration depth"
            )
        self.ignore_a = ignore_a

    def set_mci_filename(self, mci_filename: str) -> None:
        """The full path to the input file.

        E.g. ./data/my.mci
        """
        self.mci_filename = mci_filename

    def set_mcml_executable(self, mcml_executable: str) -> None:
        """The full path of the executable.

        E.g. mcmlgpu/MCML
        """
        if mcml_executable and os.path.isfile(mcml_executable):
            logger.info(f"Using provided MCML path: {mcml_executable}")
            self.mcml_executable = mcml_executable
        elif shutil.which("MCML") is not None:
            logger.info("Using system-wide MCML executable")
            self.mcml_executable = "MCML"
        else:
            raise ValueError("Could not find MCML executable")

    def run_simulation(
        self, mco_file: str = "", verbose: bool = True, timeout: float | None = None
    ) -> None:
        """This method runs a monte carlo simulation."""
        abs_mci_filename = os.path.abspath(self.mci_filename)
        args = (self.mcml_executable,)
        if self.ignore_a:
            args += ("-A",)  # type: ignore [assignment]
        if not mco_file:
            mco_file = MCO_PATH  # type: ignore [assignment]
        args += (  # type: ignore [assignment]
            "-O",
            mco_file,
        )
        args += (  # type: ignore [assignment]
            "-i",
            abs_mci_filename,
        )

        if verbose:
            mcml_exec = subprocess.Popen(args)
        else:
            mcml_exec = subprocess.Popen(args, stdout=subprocess.DEVNULL)
        mcml_exec.wait(timeout=timeout)
        if mcml_exec.returncode != 0:
            raise RuntimeError(
                f"Simulation exited with error code {mcml_exec.returncode}"
            )
