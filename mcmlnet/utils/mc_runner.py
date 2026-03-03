""""""

import os
import warnings

import numpy as np
import pandas as pd
from rich.progress import track

from mcmlnet.susi.calculate_spectra import calculate_spectrum_for_batch
from mcmlnet.utils.logging import setup_logging

logger = setup_logging(level="info", logger_name=__name__)


class RecomputeMC:
    def __init__(
        self,
        wavelengths: np.ndarray,
        nr_photons: int = 10**6,
        ignore_a: bool = True,
        mco_folder: str = "",
        verbose: bool = False,
        timeout: float | None = None,
    ) -> None:
        """
        Define basic properties

        Args:
            wavelengths: Wavelengths to simulate
            nr_photons: Number of photons to simulate
            ignore_a: Whether to ignore absorption in the simulation
            mco_folder: Folder containing the MCO files
            verbose: Whether to print verbose output
            timeout: Maximum time to wait for the simulation to complete

        Raises:
            ValueError: If wavelengths are outside the valid range [200, 2000] nm
        """
        if any(wavelengths < 200 * 10**-9) or any(wavelengths > 2000 * 10**-9):
            raise ValueError("Wavelengths must be in the range [200, 2000] *10**-9 m.")
        self.wavelengths = wavelengths
        self.nr_photons = nr_photons
        self.ignore_a = ignore_a
        self.mco_folder = mco_folder
        self.verbose = verbose
        self.timeout = timeout

        return None

    @staticmethod
    def _warn_if_batch_not_full(batch: pd.DataFrame, batch_size: int) -> None:
        """Warn if the batch size does not divide the number of samples."""
        if batch.shape[0] % batch_size != 0:
            warnings.warn(
                "Batch size does not divide the number of samples, the last "
                f"{batch.shape[0] % batch_size} samples will be ignored!",
                stacklevel=2,
            )

    @staticmethod
    def _determine_batch_range(
        batch: pd.DataFrame, batch_size: int, batch_range: tuple[int, int]
    ) -> tuple[int, int]:
        """Determine the range of batches (batch IDs) to simulate."""
        if batch_range[1] == -1:
            return (batch_range[0], batch.shape[0] // batch_size)
        return batch_range

    @staticmethod
    def generate_column_headers(columns: list[str], n_layers: int) -> list[np.ndarray]:
        """
        Returns a two-level DataFrame column header
        with the given parameter names in the specified columns.

        Args:
            columns: List of parameter names in said columns
            n_layers: Number of layers

        Returns:
            Two-level DataFrame column header
        """
        length = len(columns)
        # Repeat the layer number in the top level, parameter names in the bottom level
        top_level = (
            np.array([["layer" + str(i)] * length for i in range(n_layers)])
            .squeeze()
            .flatten()
        )
        bottom_level = np.array(columns * n_layers)

        return [top_level, bottom_level]

    def array_to_mc_sim_df(self, array: np.ndarray) -> pd.DataFrame:
        """
        Adopts DataFrame structure required similar to single_batch_from_tissue_spec
        from calculate_spectra to allow MCML simulations.

        Args:
            array: Array with physiological optical parameters

        Returns:
            pd.DataFrame: MultiIndex DataFrame with physiological optical parameters
        """
        if array.shape != (len(array), 24):
            raise AssertionError(
                "Optical parameter array has to be of shape (len, 24)."
            )

        columns = ["sao2", "vhb", "a_mie", "b_mie", "a_ray", "g", "n", "d"]
        header = self.generate_column_headers(columns, 3)

        return pd.DataFrame(array, columns=header)

    def csv_to_mc_sim_df(
        self, df: pd.DataFrame | None = None, path: str | None = None
    ) -> pd.DataFrame:
        """
        Adopts DataFrame structure required similar to single_batch_from_tissue_spec
        from calculate_spectra to allow MCML simulations.

        Loads a DataFrame from RAM or path, converts it to intermediate np.ndarray and
        returns correct MultiIndex DataFrame.

        Args:
            df: DataFrame with optical parameters
            path: Path to a CSV file containing optical parameters

        Returns:
            pd.DataFrame: MultiIndex DataFrame with optical parameters
        """
        if (df is None) and (path is None):
            raise ValueError("Missing Input: Either 'df' or 'path' have to be defined.")

        # Load df, drop unnecessary index column
        df = pd.read_csv(path) if path else df
        if "Unnamed: 0" in df.columns:  # type: ignore [union-attr]
            df.drop(["Unnamed: 0"], axis=1, inplace=True)  # type: ignore [union-attr]

        return self.array_to_mc_sim_df(df.to_numpy(dtype=np.float32))  # type: ignore [union-attr]

    def convert_batch_to_df(
        self, batch: str | pd.DataFrame | np.ndarray
    ) -> pd.DataFrame:
        """Convert various input formats to simulatable DataFrame."""
        if isinstance(batch, pd.DataFrame):
            return batch
        elif isinstance(batch, str):
            return self.csv_to_mc_sim_df(path=batch)
        elif isinstance(batch, np.ndarray):
            return self.array_to_mc_sim_df(array=batch)
        else:
            raise TypeError(
                f"Unknown type {type(batch)}. "
                "'batch' must be str, pd.DataFrame, or np.ndarray."
            )

    def _load_or_simulate_batch(
        self, batch: pd.DataFrame, save_dir: str, batch_size: int, id: int
    ) -> pd.DataFrame:
        """Load a batch from file or simulate it if not found.

        Args:
            batch: DataFrame with physiological parameters
            save_dir: Directory to save the simulated batch
            batch_size: Size of the batch to simulate
            id: Batch ID for sliced simulation

        Returns:
            pd.DataFrame: Simulated or loaded batch DataFrame
        """
        file_path = os.path.join(save_dir, f"batch_{id}.csv")
        try:
            result = pd.read_csv(file_path, header=[0, 1])
            if len(result) != batch_size:
                raise AssertionError(
                    "Batch size does not match! "
                    "Check given batch size and number of samples."
                )
        except (FileNotFoundError, TypeError):
            logger.info(f"File {file_path} not found, starting simulation...")
            result = calculate_spectrum_for_batch(
                batch[id * batch_size : (id + 1) * batch_size].reset_index(drop=True),
                self.wavelengths,
                self.nr_photons,
                mci_base_folder=self.mco_folder,
                batch_id=str(id),
                ignore_a=self.ignore_a,
                mco_file=self.mco_folder + "batch.mco",
                verbose=self.verbose,
                timeout=self.timeout,
            )
            if save_dir:
                result.to_csv(save_dir + f"batch_{id}.csv", index=False)

        return result

    def run_simulation_from_df(
        self,
        batch: str | pd.DataFrame | np.ndarray,
        save_dir: str,
        batch_size: int = 1000,
        batch_range: tuple[int, int] = (0, -1),
    ) -> np.ndarray:
        """
        Run a Monte Carlo simulation on the batch of physiological parameters loaded
        from a .csv (str), pd.DataFrame or np.ndarray and save the results to a CSV
        file. Simulation saves every batch separately and automatically resumes from
        the last missing batch if interrupted (thus assuming the same global batch).

        Args:
            batch: Batch of physiological parameters to simulate
            save_dir: Directory to save the simulation results to
            batch_size: Number of samples to simulate in each batch
            batch_range: Range of batches to simulate (start, end), where end is -1
                to simulate all batches until the end of the DataFrame.

        Returns:
            Reflectance predictions for the simulated batches
        """
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Convert batch to DataFrame if necessary
        batch = self.convert_batch_to_df(batch)
        self._warn_if_batch_not_full(batch, batch_size)
        batch_range = self._determine_batch_range(batch, batch_size, batch_range)

        # Initialize reflectance storage list
        reflectance = []

        for i in track(
            range(batch.shape[0] // batch_size), description="Running MC simulation..."
        ):
            # Skip if batch is not in range
            if i not in range(*batch_range):
                continue

            _batch = self._load_or_simulate_batch(batch, save_dir, batch_size, i)

            # Collect prediction results
            reflectance.append(_batch.reflectances.to_numpy().squeeze())

        # Log status after all batches have been simulated
        logger.info(f"Saved {i} batches of shape {_batch.shape} to {save_dir}")

        return np.concatenate(reflectance, axis=0)
