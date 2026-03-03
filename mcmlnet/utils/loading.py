"""Data loading utilities for Monte Carlo simulations."""

import os

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

from mcmlnet.utils.logging import setup_logging

load_dotenv()
logger = setup_logging(level="info", logger_name=__name__)


class SimulationDataLoader:
    """
    Unified loader for Monte Carlo simulation data from various sources.

    Handles loading from csv and parquet files.
    """

    def __init__(
        self,
        n_wavelengths: int | None = None,
        n_layers: int | None = None,
        n_physio_params: int = 8,
        n_physical_params: int = 5,
        is_physical: bool = False,
    ) -> None:
        """Initialize the simulation data loader.

        Args:
            n_wavelengths: Number of wavelengths.
            n_layers: Number of tissue layers.
            n_physio_params: Number of physiological parameters per layer.
            n_physical_params: Number of physical parameters per layer.
            is_physical: Whether data is physical (vs physiological).
        """
        self.n_wavelengths = n_wavelengths
        self.n_layers = n_layers
        self.n_physio_params = n_physio_params
        self.n_physical_params = n_physical_params
        self.is_physical = is_physical

        # Data storage
        self.simulation_df: pd.DataFrame | None = None
        self.simulations: np.ndarray | None = None
        self.simulation_tensor: torch.Tensor | None = None

    @staticmethod
    def _get_columns_by_name(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """Get columns containing a specific name."""
        cols = [col for col in df.columns if column_name in col]
        return df[cols]

    @staticmethod
    def _drop_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Drop specified columns if they exist."""
        drop_cols = [
            col
            for col in df.columns
            if any(sub in str(col) for sub in ["chb", "ua", "us"])
        ]
        return df.drop(columns=drop_cols, errors="ignore")

    @staticmethod
    def _load_from_directory(dir: str) -> pd.DataFrame:
        """Load all `.parquet` files from a directory and concatenate them."""
        batches = [
            os.path.join(dir, f)
            for f in os.listdir(dir)
            if os.path.isfile(os.path.join(dir, f)) and f.endswith(".parquet")
        ]

        if not batches:
            raise FileNotFoundError(f"No `.parquet` files found in directory: {dir}")

        return pd.concat(
            [pd.read_parquet(f) for f in batches], ignore_index=True, copy=False
        )

    def simulation_to_standard_numpy(self, data: pd.DataFrame) -> np.ndarray:
        """Convert DataFrame/Series to standard numpy array.

        Args:
            data: DataFrame containing simulation data.

        Returns:
            A numpy array with the shape
                (n_samples, n_physio_params * n_layers + n_wavelengths).

        Raises:
            AssertionError: If the data shape does not match expected dimensions.
            ValueError: If the data does not contain readable layer columns.
        """
        # Get reflectance data
        reflectances = self._get_columns_by_name(data, "reflectances")
        n_wvl = reflectances.shape[-1]

        # Validate wavelength count
        if self.n_wavelengths is None:
            self.n_wavelengths = n_wvl
        elif n_wvl != self.n_wavelengths:
            raise AssertionError("Wavelength mismatch!")

        if reflectances.ndim != 2:
            raise AssertionError("Reflectance data has unexpected shape.")
        if reflectances.shape[0] != data.shape[0]:
            raise AssertionError("Reflectance data does not match input shape.")

        # Get physiological parameters for each layer
        layer_dfs = []
        layer_num = 0

        # Collect all available layers
        while True:
            layer_name = f"layer{layer_num}"
            layer_data = self._get_columns_by_name(data, layer_name)

            if layer_data.empty:
                break

            # Drop columns unused for further computations and append to list
            layer_data = self._drop_columns(layer_data)
            layer_dfs.append(layer_data)
            layer_num += 1

        if not layer_dfs:
            raise ValueError("No layer data found in the dataset")

        # Update the number of layers
        if self.n_layers is None:
            self.n_layers = len(layer_dfs)
        elif self.n_layers != len(layer_dfs):
            raise AssertionError(
                f"Expected {self.n_layers} layers, found {len(layer_dfs)}"
            )

        # Concatenate all layers with reflectances
        simulations = pd.concat([*layer_dfs, reflectances], axis=1)
        simulations = simulations.dropna().to_numpy(dtype=np.float32)

        # Validate shape (8 parameters per layer)
        if (simulations.shape[1] - n_wvl) % self.n_physio_params != 0:
            raise AssertionError("Shape mismatch!")

        return simulations

    @staticmethod
    def seeded_shuffle_idcs(
        reference: torch.Tensor | int, seed: int = 42
    ) -> torch.Tensor:
        """Get seeded shuffled indices."""
        torch.manual_seed(seed)
        return (
            torch.randperm(len(reference))
            if isinstance(reference, torch.Tensor)
            else torch.randperm(reference)
        )

    def load_simulation_data(
        self,
        dataset_name: str = "raw/base_physio_and_physical_simulations/"
        "physiological_tissue_model_1M_photons.parquet",
        origin: str = os.environ["data_dir"],
    ) -> torch.Tensor:
        """
        Load standard Monte Carlo simulations by name.

        Args:
            dataset_name: str, name of the simulation folder or CSV file.
            origin: str | None, path to origin of CSV files.
            cache: bool, whether to cache loaded files.

        Returns:
            A tensor containing optical parameters and reflectance data.

        Raises:
            FileNotFoundError: If the dataset is not found.
        """
        self.is_physical = False

        # Create absolute path to dataset
        dir = os.path.join(origin, dataset_name)

        if os.path.isdir(dir):
            df = self._load_from_directory(dir)
        elif os.path.isfile(dir) and dir.endswith(".csv"):
            df = pd.read_csv(dir, header=[0, 1])
        elif os.path.isfile(dir) and dir.endswith(".parquet"):
            df = pd.read_parquet(dir)
        else:
            raise FileNotFoundError(f"Dataset {dataset_name} not found in {origin}.")
        logger.info(f"Unprocessed Monte Carlo database shape: {df.shape}")

        # Set wavelength count from reflectance columns
        reflectance_cols = self._get_columns_by_name(df, "reflectances")
        self.n_wavelengths = reflectance_cols.shape[-1]
        self.simulation_df = df

        # Convert to standard numpy array and torch tensor
        self.simulations = self.simulation_to_standard_numpy(self.simulation_df)
        simulation_tensor = torch.from_numpy(self.simulations)

        # Apply seeded shuffle
        self.simulation_tensor = simulation_tensor[
            self.seeded_shuffle_idcs(simulation_tensor).to(simulation_tensor.device)
        ]

        return self.simulation_tensor

    def load_physical_simulation_data(
        self,
        dataset_name: str,
        n_wavelengths: int,
        origin: str = os.environ["data_dir"],
    ) -> torch.Tensor:
        """
        Load parquet file containing physical parameters and reflectance data.

        Args:
            dataset_name: str, name of the simulation folder.
            n_wavelengths: int, number of wavelengths.
            origin: str, path to origin of parquet file.

        Returns:
            A tensor containing physical parameters and reflectance data.

        Raises:
            AssertionError: If the data shape does not match expected dimensions.
        """
        self.is_physical = True

        # Load parquet file
        simulations = pd.read_parquet(os.path.join(origin, dataset_name)).to_numpy()

        # Define and sanity-check data shape properties
        if self.n_wavelengths is None:
            self.n_wavelengths = n_wavelengths
        elif self.n_wavelengths != n_wavelengths:
            raise AssertionError("Wavelength mismatch!")
        # Case: "Short" data format (three layers only!)
        if simulations.shape[1] == 3 * self.n_physical_params + 1:
            try:
                data_array = simulations.reshape(
                    -1, self.n_wavelengths, simulations.shape[1]
                )
                reorder_pattern = [
                    0,
                    5,
                    10,  # mu_a
                    1,
                    6,
                    11,  # mu_s
                    2,
                    7,
                    12,  # g
                    3,
                    8,
                    13,  # n
                    4,
                    9,
                    14,  # d
                    15,  # reflectance
                ]
                data_array = data_array[..., reorder_pattern]
            except ValueError as err:
                raise AssertionError("Shape mismatch!") from err
        # Case: "Long" data format
        elif simulations.shape[1] % self.n_wavelengths == 0:
            n_params = self.n_physical_params
            param_shape = simulations.shape[1] // self.n_wavelengths - 1
            if param_shape % n_params != 0:
                raise AssertionError("Shape mismatch!")
            n_layers = param_shape // n_params
            if self.n_layers is None:
                self.n_layers = n_layers
            elif self.n_layers != n_layers:
                raise AssertionError("Layer mismatch!")

            # Separate reflectance from physical parameters
            reflectances = simulations[:, -self.n_wavelengths :]
            parameters = simulations[:, : -self.n_wavelengths]

            # Reshape parameters to standard 3D physical shape, e.g.:
            # (n_samples; n_wavelengths; (mu_a1, mu_a2, mu_a3,
            #   mu_s1, mu_s2, mu_s2,
            #   g1, g2, g3,
            #   n1, n2, n3,
            #   d1, d2, d3, reflectance
            # ))
            n_samples = simulations.shape[0]
            parameters = parameters.reshape(
                n_samples, self.n_wavelengths, n_params * n_layers
            )
            data_array = np.concatenate(
                (parameters, reflectances[..., np.newaxis]), axis=-1
            )
        else:
            raise AssertionError("Shape mismatch!")

        # Convert to standard numpy array and torch tensor
        self.simulations = data_array.astype(np.float32)
        simulation_tensor = torch.from_numpy(self.simulations)

        # Apply seeded shuffle
        self.simulation_tensor = simulation_tensor[
            self.seeded_shuffle_idcs(simulation_tensor).to(simulation_tensor.device)
        ]

        return self.simulation_tensor

    def load_data(
        self,
        dataset_name: str,
        n_wavelengths: int | None = None,
        origin: str = os.environ["data_dir"],
    ) -> torch.Tensor:
        """
        Load data from a dataset. Unified for both reflectance and physical parameters.

        Args:
            dataset_name: str, name of the simulation folder.
            n_wavelengths: int, number of wavelengths.
            origin: str, path to origin of parquet file.
        """
        if "physical" in dataset_name.split("/")[-1]:
            n_wvl = n_wavelengths or self.n_wavelengths
            return self.load_physical_simulation_data(dataset_name, n_wvl, origin)  # type: ignore [arg-type]
        else:
            return self.load_simulation_data(dataset_name, origin)
