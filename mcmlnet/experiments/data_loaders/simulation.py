"""Simulation data loading functionality."""

import os

import numpy as np
import pandas as pd
import torch

from mcmlnet.data_gen.camera_adaptation import adapt_simulations_to_tivita
from mcmlnet.experiments.data_loaders.config import DataConfig
from mcmlnet.experiments.data_loaders.utils import (
    subsample_data,
)
from mcmlnet.training.data_loading.preprocessing import PreProcessor
from mcmlnet.training.optimization.analytical_baselines import (
    AnalyticalModelConstants,
    compute_mu_a,
    compute_mu_s_prime,
    jacques_1999,
)
from mcmlnet.transforms.related_work_transformations import (
    LanConstants,
    ManojlovicConstants,
    ManojlovicTransformation,
    TsuiConstants,
)
from mcmlnet.utils.loading import SimulationDataLoader
from mcmlnet.utils.logging import setup_logging
from mcmlnet.utils.process_spectra import r_specular

logger = setup_logging(level="info", logger_name=__name__)


class GenericSimulationLoader:
    """Loader for generic Monte Carlo simulation data."""

    def __init__(self, specular: bool = False) -> None:
        """Initialize generic simulation loader.

        Args:
            specular: Whether to include specular reflectance.
        """
        self.specular = specular
        self.simulations = SimulationDataLoader().load_simulation_data()
        """pd.read_parquet(
            os.path.join(
                os.environ["data_dir"],
                "raw/base_physio_and_physical_simulations/physiological_tissue_model_1M_photons.parquet",
            )
        )"""

    def load_reflectances(self) -> np.ndarray:
        """Load generic Monte Carlo simulation reflectances.

        Returns:
            Array containing reflectance data with wavelength headers.
        """
        cache_path = os.path.join(os.environ["cache_dir"], "reflectances_generic.npz")

        try:
            reflectance = np.load(cache_path)["reflectances_generic"]
            logger.info("Loaded generic reflectances from cache.")
        except FileNotFoundError:
            logger.info("Loading generic reflectances from Monte Carlo simulations...")
            reflectance = self.simulations.numpy()[:, 24:]
            wavelengths = np.linspace(300, 1000, 351, endpoint=True) * 1e-9
            reflectance = adapt_simulations_to_tivita(reflectance, wavelengths)
            np.savez(cache_path, reflectances_generic=reflectance)

        if self.specular:
            refractive_indices = self.simulations.numpy()[:, [6]]
            reflectance += r_specular(1.0, refractive_indices)[:, np.newaxis]  # type: ignore [index]

        return reflectance

    def load_physiological_parameters(self) -> pd.DataFrame:
        """Load physiological parameters from generic simulations.

        Returns:
            DataFrame containing physiological parameters.
        """
        simulations = self.simulations.numpy()

        preprocessor = PreProcessor(
            val_percent=DataConfig.DEFAULT_VAL_PERCENT,
            test_percent=DataConfig.DEFAULT_TEST_PERCENT,
        )
        train_ids = preprocessor.consistent_data_split_ids(simulations, mode="train")

        layers = ["layer0", "layer1", "layer2"]
        columns = ["SaO2", "vHb", "a_Mie", "b_Mie", "a_ray", "g", "n", "d"]
        header = pd.MultiIndex.from_product([layers, columns])

        return pd.DataFrame(simulations[train_ids, :24], columns=header)

    def load_physical_parameters(self, n_wavelengths: int = 351) -> np.ndarray:
        """Load physical parameters from generic simulations.

        Args:
            n_wavelengths: Number of wavelengths to load.

        Returns:
            Array containing physical parameters.
        """
        simulations = self.simulations.numpy()

        preprocessor = PreProcessor(
            n_wavelengths=n_wavelengths,
            val_percent=DataConfig.DEFAULT_VAL_PERCENT,
            test_percent=DataConfig.DEFAULT_TEST_PERCENT,
            is_or_make_physical=True,
            log=False,
        )
        train_ids = preprocessor.consistent_data_split_ids(simulations, mode="train")
        train_data = simulations[train_ids, : 24 + n_wavelengths]

        preprocessor.n_layers = DataConfig.DEFAULT_N_LAYERS
        physical_data = preprocessor(torch.from_numpy(train_data))

        return physical_data[..., :-1].numpy()

    def load_all_data(self) -> dict[str, pd.DataFrame | np.ndarray]:
        """Load all generic simulation data.

        Returns:
            Dictionary containing reflectances and parameters.
        """
        return {
            "reflectances": self.load_reflectances(),
            "physiological_parameters": self.load_physiological_parameters(),
            "physical_parameters": self.load_physical_parameters(),
        }


class LanSimulationLoader:
    """Loader for Lan et al. (2023) simulation data."""

    def __init__(self, specular: bool = False) -> None:
        """Initialize Lan simulation loader.

        Args:
            specular: Whether to include specular reflectance.
        """
        self.specular = specular
        self.constants = LanConstants()

    def load_data(self, lhs: bool = True) -> pd.DataFrame:
        """Load Lan et al. data.

        Args:
            lhs: Whether to load LHS (Latin Hypercube Sampling) data.

        Returns:
            DataFrame containing data.
        """
        filename = "lan_lhs_2023_resim.parquet" if lhs else "lan_2023_resim.parquet"
        data_path = os.path.join(
            os.environ["data_dir"], "raw/related_work_reimplemented", filename
        )

        return pd.read_parquet(data_path).drop(columns="layer [top first]")

    def load_reflectances(self, lhs: bool = True) -> np.ndarray:
        """Load Lan et al. reflectances.

        Args:
            lhs: Whether to load LHS (Latin Hypercube Sampling) data.

        Returns:
            Array containing reflectance data.
        """
        data = self.load_data(lhs)
        reflectance = data.reflectance.to_numpy()

        if self.specular:
            reflectance += r_specular(1.0, self.constants.n)

        return reflectance

    def load_physical_parameters(self, lhs: bool = True) -> pd.DataFrame:
        """Load Lan et al. physical parameters.

        Returns:
            DataFrame containing physical parameters.
        """
        return self.load_data(lhs).layer0

    def load_all_data(self) -> dict[str, pd.DataFrame]:
        """Load all Lan simulation data.

        Returns:
            Dictionary containing reflectances and parameters.
        """
        return {
            "reflectances": self.load_reflectances(),
            "physical_parameters": self.load_physical_parameters(),
        }


class TsuiSimulationLoader:
    """Loader for Tsui et al. (2018) simulation data."""

    def __init__(self, specular: bool = False) -> None:
        """Initialize Tsui simulation loader.

        Args:
            specular: Whether to include specular reflectance.
        """
        self.specular = specular
        self.constants = TsuiConstants()

    def load_data(self) -> pd.DataFrame:
        """Load Tsui et al. data.

        Returns:
            DataFrame containing data.
        """
        return pd.read_parquet(
            os.path.join(
                os.environ["data_dir"],
                "raw/related_work_reimplemented/tsui_2018_resim.parquet",
            )
        ).drop(columns="layer [top first]")

    def load_reflectances(self) -> np.ndarray:
        """Load Tsui et al. reflectances.

        Returns:
            Array containing reflectance data.
        """
        data = self.load_data()
        reflectance = data.reflectance.to_numpy()

        if self.specular:
            reflectance += r_specular(1.0, self.constants.n)

        return reflectance

    def load_physical_parameters(self) -> pd.DataFrame:
        """Load Tsui et al. physiological parameters.

        Returns:
            DataFrame containing physiological parameters.
        """
        data = self.load_data()
        return data.loc[:, data.columns.get_level_values(0).str.startswith("layer")]

    def load_all_data(self) -> dict[str, np.ndarray | pd.DataFrame]:
        """Load all Tsui simulation data.

        Returns:
            Dictionary containing reflectances and parameters.
        """
        return {
            "reflectances": self.load_reflectances(),
            "physical_parameters": self.load_physical_parameters(),
        }


class ManojlovicSimulationLoader:
    """Loader for Manojlovic et al. (2025) simulation data."""

    def __init__(self, specular: bool = False, thick_bottom: bool = True) -> None:
        """Initialize Manojlovic simulation loader.

        Args:
            specular: Whether to include specular reflectance.
            thick_bottom: Whether to use thick bottom simulations.
        """
        self.specular = specular
        self.thick_bottom = thick_bottom
        self.constants = ManojlovicConstants()

    def generate_physiological_parameters(
        self, n_samples: int, seed: int = 0
    ) -> pd.DataFrame:
        """Generate physiological parameters for Manojlovic model.

        Args:
            n_samples: Number of samples to generate.
            seed: Random seed for reproducibility.

        Returns:
            DataFrame containing generated physiological parameters.
        """
        np.random.seed(seed)

        parameters = {
            "f_mel": np.random.uniform(
                self.constants.f_mel_range[0], self.constants.f_mel_range[1], n_samples
            ),
            "f_hb": np.random.uniform(
                self.constants.f_hb_range[0], self.constants.f_hb_range[1], n_samples
            ),
            "f_hbo2": np.random.uniform(
                self.constants.f_hbo2_range[0],
                self.constants.f_hbo2_range[1],
                n_samples,
            ),
            "f_brub": np.random.uniform(
                self.constants.f_brub_range[0],
                self.constants.f_brub_range[1],
                n_samples,
            ),
            "f_co": np.random.uniform(
                self.constants.f_co_range[0], self.constants.f_co_range[1], n_samples
            ),
            "f_coo2": np.random.uniform(
                self.constants.f_coo2_range[0],
                self.constants.f_coo2_range[1],
                n_samples,
            ),
            "a": np.random.uniform(
                self.constants.a_mie_range[0], self.constants.a_mie_range[1], n_samples
            ),
        }

        return pd.DataFrame(parameters)

    def compute_physical_parameters(self, physio_params: pd.DataFrame) -> torch.Tensor:
        """Compute Manojlovic physical parameters.

        Args:
            physio_params: Physiological parameters DataFrame.

        Returns:
            Tensor containing computed physical parameters with shape
                (n_samples, n_parameters).
            Parameters are: [mu_a_epi, scattering_epi, anisotropy_epi,
                refractive_index_epi, d_epi, mu_a_dermis, scattering_dermis,
                anisotropy_dermis, refractive_index_dermis, d_dermis]
        """
        wavelengths = torch.from_numpy(self.constants.wavelengths * 1e9)
        transformation = ManojlovicTransformation(wavelengths)

        physio_tensors = self._extract_physio_tensors(physio_params)

        epi_params = self._compute_epidermis_parameters(transformation, physio_tensors)
        dermis_params = self._compute_dermis_parameters(transformation, physio_tensors)

        return torch.cat([epi_params, dermis_params], dim=-1)

    def _extract_physio_tensors(
        self, physio_params: pd.DataFrame
    ) -> dict[str, torch.Tensor]:
        """Extract physiological parameters as tensors."""
        return {
            "f_mel": torch.from_numpy(physio_params["f_mel"].to_numpy()).float(),
            "f_hb": torch.from_numpy(physio_params["f_hb"].to_numpy()).float(),
            "f_hbo2": torch.from_numpy(physio_params["f_hbo2"].to_numpy()).float(),
            "f_brub": torch.from_numpy(physio_params["f_brub"].to_numpy()).float(),
            "f_co": torch.from_numpy(physio_params["f_co"].to_numpy()).float(),
            "f_coo2": torch.from_numpy(physio_params["f_coo2"].to_numpy()).float(),
            "a": torch.from_numpy(physio_params["a"].to_numpy()).float(),
        }

    def _compute_epidermis_parameters(
        self,
        transformation: ManojlovicTransformation,
        physio_tensors: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute epidermis layer physical parameters.

        Args:
            transformation: Manojlovic transformation object.
            physio_tensors: Dictionary containing physiological parameters as tensors.

        Returns:
            Tensor containing computed epidermis layer physical parameters.
        """
        n_samples = len(physio_tensors["f_mel"])

        # Absorption coefficient
        mu_a_epi = transformation.mu_a_epi(physio_tensors["f_mel"])

        # Scattering coefficient
        scattering = transformation.scattering(
            physio_tensors["a"],
            torch.tensor([self.constants.f_ray]),
            torch.tensor([self.constants.b_mie]),
        )

        # Common optical properties (repeated for each sample)
        anisotropy = transformation.anisotropy()[None, :].repeat(n_samples, 1)
        refractive_index = transformation.refractive_index()[None, :].repeat(
            n_samples, 1
        )

        # Thickness
        d_epi = torch.ones_like(mu_a_epi) * self.constants.d_epi

        return torch.stack(
            [mu_a_epi, scattering, anisotropy, refractive_index, d_epi], dim=-1
        )

    def _compute_dermis_parameters(
        self,
        transformation: ManojlovicTransformation,
        physio_tensors: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute dermis layer physical parameters.

        Args:
            transformation: Manojlovic transformation object.
            physio_tensors: Dictionary containing physiological parameters as tensors.

        Returns:
            Tensor containing computed dermis layer physical parameters.
        """
        n_samples = len(physio_tensors["f_mel"])

        # Absorption coefficient
        mu_a_dermis = transformation.mu_a_dermis(
            physio_tensors["f_hb"],
            physio_tensors["f_hbo2"],
            physio_tensors["f_brub"],
            physio_tensors["f_co"],
            physio_tensors["f_coo2"],
        )

        # Scattering coefficient (same as epidermis)
        scattering = transformation.scattering(
            physio_tensors["a"],
            torch.tensor([self.constants.f_ray]),
            torch.tensor([self.constants.b_mie]),
        )

        # Optical properties (repeated for each sample)
        anisotropy = transformation.anisotropy()[None, :].repeat(n_samples, 1)
        refractive_index = transformation.refractive_index()[None, :].repeat(
            n_samples, 1
        )

        # Thickness - apply thick_bottom multiplier if requested
        d_dermis = torch.ones_like(mu_a_dermis) * self.constants.d_dermis
        if self.thick_bottom:
            d_dermis *= 10

        return torch.stack(
            [mu_a_dermis, scattering, anisotropy, refractive_index, d_dermis], dim=-1
        )

    def load_data(self) -> pd.DataFrame:
        """Load Manojlovic et al. data.

        Returns:
            DataFrame containing data.
        """
        return pd.read_parquet(
            os.path.join(
                os.environ["data_dir"],
                "raw/related_work_reimplemented/manojlovic_2025_resim_thick.parquet",
            )
        ).drop(columns="Unnamed: 0")

    def load_reflectances(self) -> np.ndarray:
        """Load Manojlovic et al. reflectances.

        Returns:
            Array containing reflectance data.
        """
        if not self.thick_bottom:
            raise ValueError("Thick bottom simulations not available.")

        data = self.load_data()
        reflectance = data.iloc[:, 7:].to_numpy()

        if self.specular:
            wavelengths = torch.from_numpy(self.constants.wavelengths * 1e9)
            transformation = ManojlovicTransformation(wavelengths)
            refractive_indices = transformation.refractive_index().numpy()
            reflectance += r_specular(1.0, refractive_indices)[np.newaxis, :]  # type: ignore [index]

        return reflectance

    def load_physiological_parameters(self) -> pd.DataFrame:
        """Load Manojlovic et al. physiological parameters.

        Returns:
            DataFrame containing physiological parameters.
        """
        if not self.thick_bottom:
            raise ValueError("Thick bottom simulations not available.")

        data = self.load_data()
        return data.iloc[:, :7]

    def load_all_data(self) -> dict[str, np.ndarray | pd.DataFrame]:
        """Load all Manojlovic simulation data.

        Returns:
            Dictionary containing reflectances and parameters.
        """
        return {
            "reflectances": self.load_reflectances(),
            "physiological_parameters": self.load_physiological_parameters(),
        }


class JacquesSimulationLoader:
    """Loader for Jacques et al. (1999) simulation data."""

    def __init__(self, specular: bool = False) -> None:
        """Initialize Jacques simulation loader.

        Args:
            specular: Whether to include specular reflectance.
        """
        self.specular = specular
        self.constants = AnalyticalModelConstants()

    def generate_physiological_parameters(
        self,
        n_samples: int,
        a_mie_max: float | None = None,
        b_mie_max: float | None = None,
        f_blood_max: float | None = None,
        seed: int = 0,
    ) -> pd.DataFrame:
        """Generate physiological parameters for Jacques model.

        Args:
            n_samples: Number of samples to generate.
            a_mie_max: Maximum Mie scattering coefficient.
            b_mie_max: Maximum Mie scattering exponent.
            f_blood_max: Maximum blood volume fraction.
            seed: Random seed for reproducibility.

        Returns:
            DataFrame containing generated physiological parameters.
        """
        a_mie_max = a_mie_max or self.constants.a_mie_range[1]
        b_mie_max = b_mie_max or self.constants.b_mie_range[1]
        f_blood_max = f_blood_max or self.constants.f_blood_range[1]

        np.random.seed(seed)
        sao2 = np.random.uniform(
            self.constants.sao2_range[0], self.constants.sao2_range[1], n_samples
        )

        np.random.seed(seed + 1)
        f_blood = np.random.uniform(
            self.constants.f_blood_range[0], f_blood_max, n_samples
        )

        np.random.seed(seed + 2)
        a_mie = np.random.uniform(self.constants.a_mie_range[0], a_mie_max, n_samples)

        np.random.seed(seed + 3)
        b_mie = np.random.uniform(self.constants.b_mie_range[0], b_mie_max, n_samples)

        return pd.DataFrame(
            {
                "SaO2": sao2,
                "vHb": f_blood,
                "a_Mie": a_mie,
                "b_Mie": b_mie,
            }
        )

    def compute_physical_parameters(
        self, physio_params: pd.DataFrame, wavelengths: np.ndarray
    ) -> torch.Tensor:
        """Compute physical parameters from physiological parameters.

        Args:
            physio_params: Physiological parameters DataFrame.
            wavelengths: Wavelength array.

        Returns:
            Tensor containing computed physical parameters.
        """
        mu_a = compute_mu_a(
            physio_params.SaO2.to_numpy(), physio_params.vHb.to_numpy(), wavelengths
        )
        mu_s_prime = compute_mu_s_prime(
            physio_params.a_Mie.to_numpy(), physio_params.b_Mie.to_numpy(), wavelengths
        )

        return torch.cat([mu_a[..., None], mu_s_prime[..., None]], dim=-1)

    def load_reference_data(self, n: float) -> pd.DataFrame:
        """Load reference simulation data for Jacques model.

        Args:
            n: Refractive index value.

        Returns:
            DataFrame containing reference data.
        """
        data_path = os.path.join(
            os.environ["data_dir"],
            f"raw/related_work_reimplemented/jacques_1999_n_{n}_resim.parquet",
        )

        return pd.read_parquet(data_path)

    def compute_reflectances(
        self,
        n_samples: int,
        wavelengths: np.ndarray,
        n: float,
        a_mie_max: float | None = None,
        b_mie_max: float | None = None,
        f_blood_max: float | None = None,
        seed: int = 0,
    ) -> np.ndarray:
        """Compute reflectances using Jacques fitted semi-analytical model.

        Args:
            n_samples: Number of samples to generate.
            wavelengths: Wavelength array.
            n: Refractive index.
            a_mie_max: Maximum Mie scattering coefficient.
            b_mie_max: Maximum Mie scattering exponent.
            f_blood_max: Maximum blood volume fraction.
            seed: Random seed for reproducibility.

        Returns:
            Array containing computed reflectances.
        """
        physio_params = self.generate_physiological_parameters(
            n_samples, a_mie_max, b_mie_max, f_blood_max, seed
        )

        physical_params = self.compute_physical_parameters(physio_params, wavelengths)

        if self.specular:
            params = self.constants.jacques_1999_params_specular[n]
        else:
            params = self.constants.jacques_1999_params[n]

        reflectance = jacques_1999(
            physical_params[..., 0], physical_params[..., 1], params
        )

        return reflectance

    def load_all_data(
        self, n_samples: int, n: float = DataConfig.JACQUES_DEFAULT_REFRACTIVE_INDEX
    ) -> dict[str, pd.DataFrame | np.ndarray]:
        """Load all Jacques simulation data.

        Returns:
            Dictionary containing reflectances and parameters.
        """
        wavelengths = self.constants.wavelengths
        physio_params = self.generate_physiological_parameters(n_samples, seed=0)
        physical_params = self.compute_physical_parameters(physio_params, wavelengths)
        reflectances = self.compute_reflectances(n_samples, wavelengths, n, seed=0)

        return {
            "reflectances": reflectances,
            "physiological_parameters": physio_params,
            "physical_parameters": physical_params,
        }


class SimulationDataLoaderManager:
    """Main simulation data loader that coordinates all simulation loaders."""

    def __init__(self, specular: bool = False):
        """Initialize simulation data loader.

        Args:
            specular: Whether to include specular reflectance in all simulations.
        """
        self.specular = specular
        self.loaders = {
            "generic": GenericSimulationLoader(specular=specular),
            "lan": LanSimulationLoader(specular=specular),
            "tsui": TsuiSimulationLoader(specular=specular),
            "manoj": ManojlovicSimulationLoader(specular=specular, thick_bottom=True),
            "jacques": JacquesSimulationLoader(specular=specular),
        }

    def load_simulation_data(self) -> dict[str, torch.Tensor]:
        """Load all simulation data with consistent sizes.

        Returns:
            Dictionary mapping simulation names to tensor data.
        """
        subsample_size = DataConfig.DEFAULT_SUBSAMPLE_SIZE_TISSUE_MODEL

        manoj_raw_reflectances = self.loaders["manoj"].load_reflectances()  # type: ignore [attr-defined]
        jacques_raw_reflectances = self.loaders["jacques"].compute_reflectances(  # type: ignore [attr-defined]
            n_samples=subsample_size,
            wavelengths=self.loaders["jacques"].constants.wavelengths,  # type: ignore [attr-defined]
            n=DataConfig.JACQUES_DEFAULT_REFRACTIVE_INDEX,
        )

        simulation_data = {
            "generic_sims": subsample_data(
                self.loaders["generic"].load_reflectances(),  # type: ignore [attr-defined]
                subsample_size,
            ),
            "manoj_sims": adapt_simulations_to_tivita(
                subsample_data(manoj_raw_reflectances, subsample_size),
                self.loaders["manoj"].constants.wavelengths,  # type: ignore [attr-defined]
            ),
            "jacques_sims": adapt_simulations_to_tivita(
                subsample_data(jacques_raw_reflectances, subsample_size),
                self.loaders["jacques"].constants.wavelengths,  # type: ignore [attr-defined]
            ),
        }

        for sim_name, sim_data in simulation_data.items():
            simulation_data[sim_name] = torch.from_numpy(sim_data)

        return simulation_data

    def load_surrogate_model_data(self) -> dict[str, torch.Tensor]:
        """Load surrogate model data for regression tasks.

        Returns:
            Dictionary mapping simulation names to tensor data.
        """
        simulation_data = {
            "generic_sims": self._load_adapted_generic_data(),
            "lan_sims": self._load_adapted_lan_data(),
            "tsui_sims": self._load_adapted_tsui_data(),
            "manoj_sims": self._load_adapted_manoj_data(),
            "jacques_sims": self._load_adapted_jacques_data(),
        }

        # Convert to torch tensors
        for sim_name, sim_data in simulation_data.items():
            sim_data = subsample_data(
                sim_data, DataConfig.DEFAULT_SUBSAMPLE_SIZE_SURROGATE_MODEL
            )
            simulation_data[sim_name] = sim_data

        return simulation_data

    def _load_adapted_generic_data(self, subset_name: str = "") -> torch.Tensor:
        """Load and adapt generic regression data."""
        issi_sims = np.load(
            os.path.join(
                os.environ["data_dir"],
                f"raw/inference/reflectances_issi{subset_name}.npz",
            )
        )["reflectances_issi"]

        if self.specular:
            raise NotImplementedError(
                "Specular reflectance not implemented for ISSI surrogate model data."
            )

        wavelengths = np.linspace(300, 1000, 351, endpoint=True) * 1e-9
        return torch.from_numpy(adapt_simulations_to_tivita(issi_sims, wavelengths))

    def _load_adapted_lan_data(self) -> torch.Tensor:
        """Load and adapt Lan regression data."""
        lan_sims = np.load(
            os.path.join(
                os.environ["data_dir"],
                "raw/inference/reflectances_lan.npz",
            )
        )["reflectances_lan"]

        if self.specular:
            lan_sims += r_specular(1.0, self.loaders["lan"].constants.n)  # type: ignore [attr-defined]

        return torch.from_numpy(
            adapt_simulations_to_tivita(
                lan_sims,
                self.loaders["lan"].constants.wavelengths,  # type: ignore [attr-defined]
            )
        )

    def _load_adapted_tsui_data(self) -> torch.Tensor:
        """Load and adapt Tsui regression data."""
        tsui_sims = np.load(
            os.path.join(
                os.environ["data_dir"],
                "raw/inference/reflectances_tsui.npz",
            )
        )["reflectances_tsui"]

        if self.specular:
            tsui_sims += r_specular(1.0, self.loaders["tsui"].constants.n)  # type: ignore [attr-defined]

        return torch.from_numpy(
            adapt_simulations_to_tivita(
                tsui_sims,
                self.loaders["tsui"].constants.wavelengths,  # type: ignore [attr-defined]
            )
        )

    def _load_adapted_manoj_data(self) -> torch.Tensor:
        """Load and adapt Manojlovic regression data."""
        manoj_sims = np.load(
            os.path.join(
                os.environ["data_dir"],
                "raw/inference/reflectances_manoj.npz",
            )
        )["reflectances_manoj"]

        if self.specular:
            raise NotImplementedError(
                "Specular reflectance not implemented for Manojlovic."
            )

        return torch.from_numpy(
            adapt_simulations_to_tivita(
                manoj_sims,
                self.loaders["manoj"].constants.wavelengths,  # type: ignore [attr-defined]
            )
        )

    def _load_adapted_jacques_data(self) -> torch.Tensor:
        """Load and adapt Jacques regression data."""
        jacques_sims = self.loaders["jacques"].compute_reflectances(  # type: ignore [attr-defined]
            DataConfig.DEFAULT_SUBSAMPLE_SIZE_SURROGATE_MODEL,
            self.loaders["jacques"].constants.wavelengths,  # type: ignore [attr-defined]
            DataConfig.JACQUES_DEFAULT_REFRACTIVE_INDEX,
        )

        return torch.from_numpy(
            adapt_simulations_to_tivita(
                jacques_sims,
                self.loaders["jacques"].constants.wavelengths,  # type: ignore [attr-defined]
            )
        )

    def load_surrogate_model_ablation_data(self) -> dict[str, torch.Tensor]:
        """Load surrogate model (training data) ablation data for regression tasks.

        Returns:
            Dictionary mapping simulation names to tensor data.
        """
        simulation_data = {
            "generic_sims": self._load_adapted_generic_data(),
            "generic_sims_8400k": self._load_adapted_generic_data("_8400k"),
            "generic_sims_420k": self._load_adapted_generic_data("_420k"),
            "generic_sims_21k": self._load_adapted_generic_data("_21k"),
        }

        # Convert to torch tensors
        for sim_name, sim_data in simulation_data.items():
            sim_data = subsample_data(
                sim_data, DataConfig.DEFAULT_SUBSAMPLE_SIZE_SURROGATE_MODEL
            )
            simulation_data[sim_name] = sim_data

        return simulation_data
