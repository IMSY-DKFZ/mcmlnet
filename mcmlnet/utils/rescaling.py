"""Data rescaling utilities."""

import logging

import torch

from mcmlnet.utils.logging import setup_logging
from mcmlnet.utils.tensor import TensorType, tensor_conversion_decorator

setup_logging(level="info")
logger = logging.getLogger(__name__)


class DataRescaler:
    """
    Rescales data using different methods,
    such as Gaussian normalization or min-max scaling.
    """

    def __init__(
        self,
        scale: str | None,
        eps: float = 1e-8,
        norm_1: torch.Tensor | None = None,
        norm_2: torch.Tensor | None = None,
    ) -> None:
        """Initialize the DataRescaler.

        Args:
            scale: The scaling method to use
                ("min-max" ([0,1]), "min-max-sym" ([-1,1]), or "z-score").
            eps: A small value to avoid division by zero. Defaults to 1e-8.
            norm_1: The first normalization tensor. Defaults to None.
            norm_2: The second normalization tensor. Defaults to None.

        Raises:
            ValueError: If the scale is not one of the supported methods.
        """
        self.norm_1 = norm_1
        self.norm_2 = norm_2
        self.scale = scale
        self.eps = eps

        # Initialize cache dictionaries
        self._cached_norms: dict[
            str,
            tuple[torch.Tensor, torch.Tensor],
        ] = {}
        self._cached_eps: dict[str, torch.Tensor] = {}

        # Validate scale parameter
        self.valid_scales = ["min-max", "min-max-sym", "z-score", "None", None]
        if scale not in self.valid_scales:
            raise ValueError(
                f"Invalid scale '{scale}'. Choose from: {self.valid_scales}"
            )

    @staticmethod
    def _validate_inputs(
        data: torch.Tensor, reference_ids: list[int] | None = None
    ) -> None:
        """Validate input data and reference indices.

        Args:
            data: Data to validate.
            reference_ids: Indices to use for fitting (to avoid data leakage).

        Raises:
            TypeError: If data is not a torch.Tensor.
            ValueError: If data is not 2D or 3D, or if reference_ids are out of bounds.
        """
        if not isinstance(data, torch.Tensor):
            raise TypeError("Data must be a torch.Tensor")

        if data.numel() == 0:
            raise ValueError("Data is empty")

        if data.ndim not in [2, 3]:
            raise ValueError(f"Data must be 2D or 3D, got {data.ndim}D")

        if reference_ids is not None:
            if not all(0 <= idx < data.shape[0] for idx in reference_ids):
                raise ValueError("Reference indices out of bounds")
            if len(reference_ids) == 0:
                raise ValueError("Reference indices cannot be empty")

    @staticmethod
    def _expand_dims(
        data: torch.Tensor,
    ) -> tuple[None, slice] | tuple[None, None, slice]:
        """Expand dimensions for rescaling based on data shape."""
        return (None, None, slice(None)) if data.ndim == 3 else (None, slice(None))

    @staticmethod
    def _dims(data: torch.Tensor) -> int | tuple[int, int]:
        """Get the dimensions to rescale based on data shape."""
        return (0, 1) if data.ndim == 3 else 0

    def _check_scaling_params(self) -> None:
        """Check if scaling parameters are available.

        Raises:
            RuntimeError: If scaling parameters are not computed for the current scale.
        """
        if self.scale not in self.valid_scales:
            raise ValueError(
                f"Invalid scale '{self.scale}'. Choose from: {self.valid_scales}"
            )
        if self.scale not in [None, "None"] and (
            self.norm_1 is None or self.norm_2 is None
        ):
            raise RuntimeError(
                f"Scaling parameters not computed for scale '{self.scale}'. "
                "Call fit() first."
            )

    def _get_cached_norms(
        self,
        data: torch.Tensor,
        expand_dims: tuple[None, slice] | tuple[None, None, slice],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cached normalization tensors for the current device."""
        cache_key = f"{data.device}_{data.dtype}_{expand_dims}"

        if cache_key in self._cached_norms:
            return self._cached_norms[cache_key]

        # Move norms to device and cache
        norm_1 = self.norm_1[expand_dims].to(data.device)  # type: ignore [index]
        norm_2 = self.norm_2[expand_dims].to(data.device)  # type: ignore [index]

        self._cached_norms[cache_key] = (norm_1, norm_2)

        # Clean up old cache entries
        if len(self._cached_norms) > 4:
            oldest_key = next(iter(self._cached_norms))
            del self._cached_norms[oldest_key]

        return norm_1, norm_2

    def _get_cached_eps(self, data: torch.Tensor) -> torch.Tensor:
        """Get cached epsilon for the current device."""
        cache_key = f"{data.device}_{data.dtype}"

        if cache_key in self._cached_eps:
            return self._cached_eps[cache_key]

        eps = torch.tensor(self.eps, device=data.device)
        self._cached_eps[cache_key] = eps

        return eps

    def _compute_z_score_scaling(
        self, data: torch.Tensor, reference_ids: list[int], dims: int | tuple[int, int]
    ) -> None:
        """Compute the mean and standard deviation for z-score scaling."""
        self.norm_1 = data[reference_ids].mean(dim=dims, keepdim=False)
        self.norm_2 = data[reference_ids].std(dim=dims, keepdim=False)

    def _apply_z_score_scaling(
        self,
        data: torch.Tensor,
        expand_dims: tuple[None, slice] | tuple[None, None, slice],
    ) -> torch.Tensor:
        """Apply z-score scaling to data."""
        norm_1, norm_2 = self._get_cached_norms(data, expand_dims)
        eps = self._get_cached_eps(data)

        return (data - norm_1) / torch.clamp(norm_2, min=eps)

    def _compute_min_max_scaling(
        self, data: torch.Tensor, reference_ids: list[int], dims: int | tuple[int, int]
    ) -> None:
        """Compute the min and max for min-max scaling."""
        self.norm_1 = data[reference_ids].amin(dim=dims, keepdim=False)
        self.norm_2 = data[reference_ids].amax(dim=dims, keepdim=False) - self.norm_1

    def _apply_min_max_scaling(
        self,
        data: torch.Tensor,
        expand_dims: tuple[None, slice] | tuple[None, None, slice],
    ) -> torch.Tensor:
        """Apply min-max scaling to data."""
        norm_1, norm_2 = self._get_cached_norms(data, expand_dims)
        eps = self._get_cached_eps(data)

        return (data - norm_1) / torch.clamp(norm_2, min=eps)

    def fit(
        self, data: torch.Tensor, reference_ids: list[int] | None = None
    ) -> "DataRescaler":
        """Fit the rescaler to data (compute normalization parameters).

        Args:
            data: Data to fit the rescaler on
            reference_ids: Indices to use for fitting (to avoid data leakage)

        Returns:
            Self for method chaining
        """
        self._validate_inputs(data, reference_ids)
        data = data.float()

        if reference_ids is None:
            reference_ids = list(range(data.shape[0]))

        if self.scale in ["z-score"]:
            self._compute_z_score_scaling(data, reference_ids, self._dims(data))
        elif self.scale in ["min-max", "min-max-sym"]:
            self._compute_min_max_scaling(data, reference_ids, self._dims(data))

        return self

    def transform(
        self, data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform data using fitted parameters."""
        return self(data), self.norm_1, self.norm_2

    @tensor_conversion_decorator
    def fit_transform(
        self, data: TensorType, reference_ids: list[int] | None = None
    ) -> tuple[TensorType, TensorType, TensorType]:
        """Fit and transform data in one step."""
        return self.fit(data, reference_ids).transform(data)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        Convenience function to apply rescaling
        to (optical) parameters, reflectance, etc.

        Args:
            data: Data to be rescaled.

        Returns:
            The rescaled data.
        """
        self._validate_inputs(data)
        self._check_scaling_params()

        if self.scale == "z-score":
            data = self._apply_z_score_scaling(data, self._expand_dims(data))
        elif self.scale in ["min-max", "min-max-sym"]:
            data = self._apply_min_max_scaling(data, self._expand_dims(data))
            if self.scale == "min-max-sym":
                data -= 0.5
                data *= 2

        return data

    @tensor_conversion_decorator
    def inverse_transform(self, data: TensorType) -> TensorType:
        """Apply inverse transformation to rescaled data.

        Args:
            data: Rescaled data to be transformed back to original scale

        Returns:
            Data in original scale

        Raises:
            ValueError: If the scale is not one of the supported methods.
        """
        self._check_scaling_params()
        data = data.float()
        expand_dims = self._expand_dims(data)

        if self.scale == "z-score":
            return data * self.norm_2[expand_dims].to(data.device) + self.norm_1[  # type: ignore [index]
                expand_dims
            ].to(data.device)
        elif self.scale == "min-max-sym":
            # Reverse symmetrical scaling
            data = data / 2 + 0.5
            return data * self.norm_2[expand_dims].to(data.device) + self.norm_1[  # type: ignore [index]
                expand_dims
            ].to(data.device)
        elif self.scale == "min-max":
            return data * self.norm_2[expand_dims].to(data.device) + self.norm_1[  # type: ignore [index]
                expand_dims
            ].to(data.device)

        return data
