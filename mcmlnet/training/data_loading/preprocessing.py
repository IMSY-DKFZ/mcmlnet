"""Data preprocessing functionality."""

import numpy as np
import torch
import torch.jit
from sklearn.model_selection import KFold

from mcmlnet.transforms.physiological import PhysiologicalToPhysicalTransformer
from mcmlnet.utils.loading import SimulationDataLoader
from mcmlnet.utils.logging import setup_logging
from mcmlnet.utils.rescaling import DataRescaler
from mcmlnet.utils.tensor import TensorType

logger = setup_logging(level="info", logger_name=__name__)


def subsample_wavelengths(simulations: TensorType, n_wvl: int) -> TensorType:
    """Subsample wavelengths from 351 to n_wvl.

    Args:
        simulations: Data tensor, with reflectance data as last columns.
        n_wvl: Amount of wavelengths.

    Returns:
        Subsampled data tensor.

    Raises:
        ValueError: If n_wvl is not positive or greater than base_n_wvl.
        TypeError: If simulations is not a valid tensor type.
    """
    # Define base amount of wavelengths and reduction factor
    base_n_wvl = 351
    if not isinstance(n_wvl, int) or n_wvl <= 0 or n_wvl > base_n_wvl:
        raise ValueError(
            f"n_wvl must be a positive integer <= {base_n_wvl}, got {n_wvl}"
        )

    if not isinstance(simulations, torch.Tensor | np.ndarray):
        raise TypeError(
            f"simulations must be torch.Tensor or np.ndarray, got {type(simulations)}"
        )

    reduction_factor = base_n_wvl // n_wvl
    logger.warning(f"Subsampling wavelengths from {base_n_wvl} to {n_wvl}!")

    # Subsample wavelengths by subsampling the last -base_n_wvl columns
    reflectances = simulations[..., -base_n_wvl::reduction_factor][..., :n_wvl]

    # Combine physiological and reflectance data
    if isinstance(simulations, torch.Tensor):
        return torch.cat([simulations[..., :-base_n_wvl], reflectances], dim=-1)
    else:
        return np.c_[simulations[..., :-base_n_wvl], reflectances]


class PreProcessor:
    """Data preprocessor for MC simulation data."""

    def __init__(
        self,
        # Core parameters (most commonly used)
        dataset_name: str = "reflectance",
        n_wavelengths: int = 100,
        val_percent: float = 0.1,
        test_percent: float = 0.2,
        # Data transformation parameters
        is_or_make_physical: bool = True,
        log: bool = False,
        n_layers: int = 3,
        # Normalization parameters
        refl_range: str = "None",
        param_range: str | None = None,
        # Cross-validation parameters (optional)
        kfolds: int | None = None,
        fold: int | None = None,
        # Advanced parameters (rarely used)
        log_intensity: bool = False,
        n_pca_comp: int | None = None,
        batch_size: int | None = None,
        wavelengths: np.ndarray | None = None,
        norm_1: torch.Tensor | None = None,
        norm_2: torch.Tensor | None = None,
        pca_component_list: list[list[float]] | None = None,
        pca_mean_list: list[list[float]] | None = None,
    ):
        """Initialize the preprocessor.

        Args:
            # Core parameters
            dataset_name: Name of the dataset to load.
            n_wavelengths: Number of wavelengths to use.
            val_percent: Validation data percentage (0-1).
            test_percent: Test data percentage (0-1).

            # Data transformation parameters
            is_or_make_physical: Whether to transform physiological
                to physical parameters.
            log: Whether to apply log10 transformation to mu_a, mu_s, and d.
            n_layers: Number of layers for log transformation.

            # Normalization parameters
            refl_range: Reflectance normalization method ("None", "asym", etc.).
            param_range: Parameter normalization method.

            # Cross-validation parameters
            kfolds: Number of k-folds for cross-validation.
            fold: Current fold for cross-validation.

            # Advanced parameters
            log_intensity: Whether to apply log intensity transformation.
            n_pca_comp: Number of PCA components.
            batch_size: Batch size for processing.
            wavelengths: Custom wavelengths array.
            norm_1: Pre-computed normalization parameter 1.
            norm_2: Pre-computed normalization parameter 2.
            pca_component_list: Pre-computed PCA components.
            pca_mean_list: Pre-computed PCA means.

        Raises:
            ValueError: If parameters are invalid.
        """
        # Validate core parameters
        if not isinstance(n_wavelengths, int) or n_wavelengths <= 0:
            raise ValueError(
                f"n_wavelengths must be positive integer, got {n_wavelengths}"
            )

        if (
            not isinstance(val_percent, int | float)
            or val_percent < 0
            or val_percent > 1
        ):
            raise ValueError(f"val_percent must be between 0 and 1, got {val_percent}")

        if (
            not isinstance(test_percent, int | float)
            or test_percent < 0
            or test_percent > 1
        ):
            raise ValueError(
                f"test_percent must be between 0 and 1, got {test_percent}"
            )

        if val_percent + test_percent >= 1.0:
            raise ValueError(
                "val_percent + test_percent must be less than 1, "
                f"got {val_percent + test_percent}"
            )

        if not isinstance(n_layers, int) or n_layers <= 0:
            raise ValueError(f"n_layers must be positive integer, got {n_layers}")

        # Store core parameters
        self.dataset_name = dataset_name
        self.n_wavelengths = n_wavelengths
        self.val_percent = val_percent
        self.test_percent = test_percent

        # Store transformation parameters
        self.is_or_make_physical = is_or_make_physical
        self.log_parameters = log
        self.n_layers = n_layers

        # Store normalization parameters
        self.refl_range = refl_range
        self.param_range = param_range

        # Store cross-validation parameters
        self.kfolds = kfolds
        self.fold = fold

        # Store advanced parameters
        self.log_intensity = log_intensity
        self.n_pca_comp = n_pca_comp
        self.batch_size = batch_size
        self.wavelengths = wavelengths
        self.norm_1 = norm_1
        self.norm_2 = norm_2
        self.pca_component_list = pca_component_list or []
        self.pca_mean_list = pca_mean_list or []

        # Initialize data loader
        self.data_loader = SimulationDataLoader(
            n_wavelengths=self.n_wavelengths, n_layers=self.n_layers
        )

        # Initialize physical parameter transformation
        self.physical_transformation = PhysiologicalToPhysicalTransformer(
            n_wavelengths=self.n_wavelengths,
            n_layers=self.n_layers,
        )

    def consistent_data_split_ids(
        self,
        data: TensorType,
        mode: str,
        kfolds: int | None = None,
        fold: int | None = None,
    ) -> list[int]:
        """Create consistent data splits for train/val/test or k-fold cross-validation.

        Args:
            data: The data to split.
            mode: The split mode ('train', 'val', 'test').
            kfolds: Number of k-folds for cross-validation (optional).
            fold: Current fold for k-fold cross-validation (optional).

        Returns:
            List of indices for the requested data split.

        Raises:
            NotImplementedError: If mode is not 'train', 'val', or 'test'.
            AssertionError: If k-fold parameters are inconsistent.
        """
        # Calculate split sizes
        total_samples = len(data)
        n_val_samples = int(total_samples * self.val_percent)
        n_test_samples = int(total_samples * self.test_percent)
        n_train_samples = int(
            total_samples * (1 - (self.val_percent + self.test_percent))
        )

        logger.info(
            f"Dataset sizes: {total_samples}, {n_train_samples}, {n_val_samples}, "
            f"{n_test_samples} (total/train/val/test)"
        )

        # Get base indices for each mode
        if mode == "train":
            # Training data comes after validation and test data
            split_indices = list(range(n_val_samples + n_test_samples, total_samples))
            if kfolds is None:
                logger.warning("No k-folds specified! Using single training split.")

        elif mode == "val":
            split_indices = list(range(n_val_samples))

        elif mode == "test":
            split_indices = list(range(n_val_samples, n_val_samples + n_test_samples))

        else:
            raise NotImplementedError(
                f"Mode {mode} not implemented! Use 'train', 'val' or 'test' instead."
            )

        # Apply k-fold cross-validation if requested (only for training data split)
        if kfolds is not None and mode == "train":
            if fold is None:
                raise AssertionError(
                    "Fold must be specified for k-fold cross-validation!"
                )
            if fold >= kfolds:
                raise AssertionError(
                    f"Fold {fold} not found in {kfolds} splits (0-indexed)."
                )

            kf = KFold(n_splits=kfolds, shuffle=False)
            for i, (train_index, _) in enumerate(kf.split(split_indices)):
                if i == fold:
                    split_indices = [split_indices[j] for j in train_index]
                    break

        return split_indices

    def _preprocess_reflectances(
        self, data_tensor: torch.Tensor, train_ids: list[int]
    ) -> torch.Tensor:
        """Preprocess reflectance data.

        Args:
            data_tensor: Input data tensor.
            train_ids: Training indices.

        Returns:
            Preprocessed data tensor.
        """
        return DataRescaler(self.refl_range).fit_transform(data_tensor, train_ids)

    def _preprocess_parameters(
        self,
        data_slice: torch.Tensor,
    ) -> torch.Tensor:
        """Preprocess physiological parameters.

        Args:
            data_slice: Input data slice.

        Returns:
            Preprocessed data slice.
        """
        # Transform physiological to physical parameters
        if data_slice.ndim == 2:
            params = self.physical_transformation.transform_hb_format(
                data_slice[:, : -self.n_wavelengths]
            )
            data_slice = torch.cat(
                [params, data_slice[:, -self.n_wavelengths :].unsqueeze(2)], dim=-1
            )

        # Apply log transformation to physical parameters
        if self.log_parameters:
            data_slice[..., :-1] = self._apply_log10_to_mua_mus_d(data_slice[..., :-1])

        return data_slice

    def _process_batches(self, train_ids: list[int]) -> range:
        """Process data in batches for memory efficiency.

        Args:
            train_ids: Training indices.

        Returns:
            Range of batch indices.
        """
        if self.batch_size is None:
            return range(0, len(train_ids), len(train_ids))
        else:
            return range(0, len(train_ids), self.batch_size)

    @staticmethod
    @torch.jit.script  # type: ignore
    def _jit_apply_log10_to_mua_mus_d(
        parameter_tensor: torch.Tensor, n_layers: int
    ) -> torch.Tensor:
        """JIT-compiled log10 transformation."""
        # Process physical parameters (n_samples, n_wavelengths, n_parameters)
        # Apply log10 to mu_a (index 0:n_layers), mu_s (index n_layers:2*n_layers),
        # and d (index 4 * n_layers:5 * n_layers)
        mu_a_indices = list(range(n_layers))
        mu_s_indices = list(range(n_layers, 2 * n_layers))
        d_indices = list(range(4 * n_layers, 5 * n_layers))
        # Combine all indices into one operation
        all_indices = mu_a_indices + mu_s_indices + d_indices
        parameter_tensor[:, :, all_indices] = torch.log10(
            parameter_tensor[:, :, all_indices] + 1e-8
        )

        return parameter_tensor

    def _apply_log10_to_mua_mus_d(self, parameter_tensor: torch.Tensor) -> torch.Tensor:
        """Apply log10 transformation to mu_a, mu_s, and d (physical) parameters.

        Args:
            parameter_tensor: The physical parameter tensor.

        Returns:
            Transformed data tensor.
        """
        if not isinstance(parameter_tensor, torch.Tensor):
            raise TypeError(
                f"parameter_tensor must be torch.Tensor, got {type(parameter_tensor)}"
            )
        if not parameter_tensor.ndim == 3:
            raise ValueError(
                f"parameter_tensor must be 3D, got {parameter_tensor.ndim}D"
            )
        if not (parameter_tensor.shape[2]) % self.n_layers == 0:
            raise ValueError(
                "Number of parameters must be divisible by n_layers, "
                f"got {parameter_tensor.shape[2]}"
            )

        return self._jit_apply_log10_to_mua_mus_d(parameter_tensor, self.n_layers)

    def _compute_normalization(
        self,
        data_slice: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Compute normalization parameters.

        Args:
            data_slice: Input data slice.

        Returns:
            Tuple of normalization parameters.
        """
        # Apply basic rescaling operations, assumes data_slice is training data only!
        if self.is_or_make_physical:
            _, norm_1, norm_2 = DataRescaler(self.param_range).fit_transform(
                data_slice[..., :-1], None
            )
        else:
            _, norm_1, norm_2 = DataRescaler(self.param_range).fit_transform(
                data_slice[:, : -self.n_wavelengths],
                None,
            )
        return norm_1, norm_2

    def _aggregate_normalization(
        self,
        norm_1: list[torch.Tensor],
        norm_2: list[torch.Tensor],
        n_samples: list[int],
    ) -> None:
        """Aggregate normalization parameters across batches.

        Args:
            norm_1: List of norm_1 tensors.
            norm_2: List of norm_2 tensors.
            n_samples: List of sample counts.
        """
        if not norm_1 or not norm_2:
            self.norm_1 = None
            self.norm_2 = None
            return
        elif self.param_range == "z-score":
            total_samples = sum(n_samples)
            overall_mean = (
                torch.stack(
                    [m * n for m, n in zip(norm_1, n_samples, strict=False)], dim=0
                ).sum(dim=0)
                / total_samples
            )
            overall_var = (
                torch.stack(
                    [
                        n * (s**2 + (m - overall_mean) ** 2)
                        for m, s, n in zip(norm_1, norm_2, n_samples, strict=False)
                    ],
                    dim=0,
                ).sum(dim=0)
                / total_samples
            )
            overall_std = torch.sqrt(overall_var)
            self.norm_1, self.norm_2 = overall_mean, overall_std
        else:
            raise NotImplementedError(
                f"Normalization method {self.param_range} not implemented"
            )

    def apply_normalization_to_parameters(
        self,
        data_slice: torch.Tensor,
    ) -> torch.Tensor:
        """Apply normalization to parameters.

        Args:
            data_slice: Input data slice.

        Returns:
            Normalized data slice.
        """
        if self.is_or_make_physical:
            # Apply physiological parameter preprocessing
            data_slice = self._preprocess_parameters(data_slice)

            # Apply normalization
            rescaler = DataRescaler(
                self.param_range, norm_1=self.norm_1, norm_2=self.norm_2
            )
            data_slice[..., :-1] = rescaler.transform(data_slice[..., :-1])[0]

            # Apply log_intensity
            if self.log_intensity:
                data_slice[..., -1] = torch.log10(data_slice[..., -1])

        else:
            # Apply normalization
            rescaler = DataRescaler(
                self.param_range, norm_1=self.norm_1, norm_2=self.norm_2
            )
            data_slice[:, : -self.n_wavelengths] = rescaler.transform(
                data_slice[:, : -self.n_wavelengths]
            )[0]

            # Apply log_intensity
            if self.log_intensity:
                data_slice[:, -self.n_wavelengths :] = torch.log10(
                    data_slice[:, -self.n_wavelengths :]
                )

        return data_slice

    def fit(self) -> torch.Tensor:
        """Fit the preprocessor to the data.

        Returns:
            The loaded but unprocessed data.
        """
        # Load data and get train ids for normalization
        data_tensor = self.data_loader.load_data(self.dataset_name, self.n_wavelengths)
        train_ids = self.consistent_data_split_ids(
            data_tensor, "train", self.kfolds, self.fold
        )

        # Compute normalization
        norm_1_list = []
        norm_2_list = []
        n_samples_list = []

        for start_idx in self._process_batches(train_ids):
            # Process data in batches
            end_idx = min(
                start_idx + (self.batch_size or len(train_ids)), len(train_ids)
            )
            batch_ids = train_ids[start_idx:end_idx]
            batch_data = data_tensor[batch_ids]

            # Transform physiological to physical parameters
            # and apply log transformation
            if self.is_or_make_physical:
                batch_data = self._preprocess_parameters(batch_data)

            norm_1, norm_2 = self._compute_normalization(batch_data)
            if norm_1 is not None:
                norm_1_list.append(norm_1)
                norm_2_list.append(norm_2)
                n_samples_list.append(len(batch_ids))

        self._aggregate_normalization(norm_1_list, norm_2_list, n_samples_list)

        return data_tensor

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply previously initialized parameter pre-processing steps."""
        return self.apply_normalization_to_parameters(data)


def process_2d_3d_data(
    data: torch.Tensor,
    index: int | list[int] | torch.Tensor,
    preprocessor: PreProcessor,
) -> torch.Tensor:
    """Process the data tensor based on its dimensions.

    Args:
        data: The data tensor.
        index: The index to access the data.
        preprocessor: The preprocessor function to apply to the data.

    Returns:
        The processed data tensor.

    Raises:
        ValueError: If data tensor is not 2D or 3D.
    """
    if data.ndim not in [2, 3]:
        raise ValueError(f"Data tensor must be 2D or 3D, got {data.ndim}D")

    if isinstance(index, torch.Tensor) and index.ndim == 0:
        index = index.item()

    if isinstance(index, torch.Tensor) and index.ndim == 1:
        index = index.tolist()

    # Validate index
    if isinstance(index, int):
        if index < 0 or index >= data.shape[0]:
            raise ValueError(
                f"Index {index} out of bounds for tensor with {data.shape[0]} samples"
            )
    elif isinstance(index, list):
        if not all(isinstance(i, int) for i in index):
            raise TypeError("All elements in index list must be integers")
        if not all(0 <= i < data.shape[0] for i in index):
            raise ValueError(
                f"Some indices in {index} out of bounds "
                f"for tensor with {data.shape[0]} samples"
            )
    else:
        raise TypeError(
            f"index must be int, list[int], or torch.Tensor, got {type(index)}"
        )

    return preprocessor(data[index] if isinstance(index, list) else data[[index]])


def set_deepest_layer_to_zero(data: torch.Tensor, thick_layer: bool) -> torch.Tensor:
    """Set the deepest layer to zero to avoid artifacts.

    Args:
        data: The data tensor.
        thick_layer: Whether the deepest layer is very thick.

    Returns:
        The data tensor with the deepest layer set to zero if thick_layer is True.

    Raises:
        TypeError: If data is not a torch.Tensor or thick_layer is not a bool.
    """
    if not isinstance(data, torch.Tensor):
        raise TypeError(f"data must be torch.Tensor, got {type(data)}")

    if not isinstance(thick_layer, bool):
        raise TypeError(f"thick_layer must be bool, got {type(thick_layer)}")

    if thick_layer and data.shape[-1] >= 2:
        data[..., -2] = 0
    return data


def collate_variable_tensors_old(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate function to concatenate variable-length tensors in a batch.

    Args:
        batch: A tuple containing input and target tensors.

    Returns:
        A tuple containing the concatenated input and target tensors.
    """
    inputs, targets = zip(*batch, strict=False)
    return torch.cat(inputs), torch.cat(targets)


def collate_variable_tensors(
    data: list[torch.Tensor],
    preprocessor: PreProcessor,
    thick_deepest_layer: bool,
    n_params: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function to preprocess physiological data batches in an aggregated manner.

    Args:
        data: A tensor containing the physiological data.
        preprocessor: A preprocessor function to apply to the data.
        thick_deepest_layer: A boolean indicating whether to set
            the deepest tissue layer thickness to zero.
        n_params: The number of parameters to extract from the data.

    Returns:
        A tuple containing the concatenated input and target tensors.
    """
    processed = preprocessor(torch.stack(data))
    processed = set_deepest_layer_to_zero(processed, thick_deepest_layer)

    params = processed[..., :n_params].reshape(-1, n_params)
    reflectance = processed[..., n_params:].reshape(-1, 1).squeeze()

    return params, reflectance
