"""Physiological to physical parameter transformations."""

import numpy as np
import torch
from dotenv import load_dotenv

from mcmlnet.constants import (
    NM_TO_M,
    RAYLEIGH_EXPONENT,
    REFERENCE_WAVELENGTH_NM,
)
from mcmlnet.utils.haemoglobin_extinctions import (
    get_haemoglobin_extinction_coefficients,
)
from mcmlnet.utils.logging import setup_logging

load_dotenv()
logger = setup_logging(level="info", logger_name=__name__)


class PhysiologicalToPhysicalTransformer:
    def __init__(
        self,
        n_wavelengths: int,
        n_layers: int = 3,
        cHb: float | torch.Tensor = 150.0,
        eps: float = 1e-8,
        wavelengths: np.ndarray | None = None,
    ) -> None:
        """Transform physiological parameters to physical parameters.

        Args:
            n_wavelengths: Number of wavelengths
            n_layers: Number of tissue layers (must be >= 1)
            cHb: Hemoglobin concentration in g/L
            eps: Small value to avoid division by zero
            wavelengths: Wavelengths array in nanometers.
                If None, generates default array.

        Raises:
            ValueError: If n_layers < 1 or wavelengths shape mismatch or
                if wavelengths are outside valid range.
        """
        if n_layers < 1:
            raise ValueError(f"Number of layers must be >= 1, got {n_layers}")

        if wavelengths is not None and len(wavelengths) != n_wavelengths:
            raise ValueError(
                f"Wavelengths length {len(wavelengths)} "
                f"!= n_wavelengths {n_wavelengths}"
            )

        self.n_wavelengths = n_wavelengths
        self.n_layers = n_layers
        self.cHb = cHb
        self.eps = eps

        # Generate default wavelengths if not provided
        if wavelengths is None:
            wavelengths = self._get_wavelengths(n_wavelengths)

        # Load extinction coefficients
        eHbO2, eHb, bounds = get_haemoglobin_extinction_coefficients()

        # Validate wavelengths are within bounds
        if np.any(wavelengths * NM_TO_M < bounds[0]) or np.any(
            wavelengths * NM_TO_M > bounds[1]
        ):
            raise ValueError(
                "Some wavelengths outside valid range "
                f"[{bounds[0] / NM_TO_M:.1f}, {bounds[1] / NM_TO_M:.1f}] nm"
            )

        self.eHbO2 = (
            torch.from_numpy(eHbO2(wavelengths * NM_TO_M))
            .view(1, -1, 1)
            .expand(-1, -1, self.n_layers)
        )
        self.eHb = (
            torch.from_numpy(eHb(wavelengths * NM_TO_M))
            .view(1, -1, 1)
            .expand(-1, -1, self.n_layers)
        )

        # Molecular weight of hemoglobin
        self.gmw_hb = 64500  # g/mol
        self.prefactor = torch.log(torch.Tensor([10.0])) * self.cHb / self.gmw_hb

        # Store wavelengths
        self.wavelengths = (
            torch.from_numpy(wavelengths).view(1, -1, 1).expand(-1, -1, self.n_layers)
        )

        # Create cache
        self.layer_base_ids = torch.tensor(
            [i * 8 for i in range(self.n_layers)], dtype=torch.long
        )
        self._cached_tensors: dict[
            str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}

    @staticmethod
    def _min_max(x: torch.Tensor, name: str = "") -> None:
        """Log min and max values of a tensor."""
        logger.info(f"{name} min: {x.min():.4f}, max: {x.max():.4f}")

    def _get_wavelengths(self, n_wavelengths: int) -> np.ndarray:
        """Get wavelength array.

        Args:
            n_wavelengths: Number of wavelengths

        Returns:
            Wavelength array

        Raises:
            ValueError: If the number of wavelengths is unsupported.
        """
        if n_wavelengths == 100:
            return np.linspace(
                500, 1000, n_wavelengths, endpoint=True, dtype=np.float64
            )
        elif n_wavelengths == 351:
            return np.linspace(
                300, 1000, n_wavelengths, endpoint=True, dtype=np.float64
            )
        else:
            reduction_factor = 351 // n_wavelengths
            return np.linspace(300, 1000, 351)[::reduction_factor][:n_wavelengths]

    def _cache_constants(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cache constants to the same device and dtype."""
        cache_key = f"{dtype}_{device}"

        if cache_key in self._cached_tensors:
            return self._cached_tensors[cache_key]

        # Move constants to the same device and dtype
        eHbO2 = self.eHbO2.to(dtype=dtype, device=device)
        eHb = self.eHb.to(dtype=dtype, device=device)
        wavelengths = self.wavelengths.to(dtype=dtype, device=device)
        self._cached_tensors[cache_key] = (eHbO2, eHb, wavelengths)

        # Clean up old cache entries if too many
        if len(self._cached_tensors) > 4:  # Keep only recent device/dtype combinations
            # Remove oldest entry
            oldest_key = next(iter(self._cached_tensors))
            del self._cached_tensors[oldest_key]

        return self._cached_tensors[cache_key]

    def transform_hb_format(self, physio: torch.Tensor) -> torch.Tensor:
        """Transform physiological parameters to physical parameters.

        Args:
            physio: Physiological parameters with shape (n_samples, n_layers * 8)
                   Parameters per layer: [sao2, vhb, a_mie, b_mie, a_ray, g, n, d]

        Returns:
            Physical parameters with shape (n_samples, n_wavelengths * n_layers * 5)
            Parameters per layer: [mu_a, mu_s, g, n, d], grouped by parameter type i.e.
            [mu_a_layer0, mu_a_layer1, ..., mu_s_layer0, ..., d_layerN]

        Raises:
            ValueError: If input shape is incorrect
        """
        # Validate input shape - 8 parameters per layer (fixed cHb only!)
        expected_features = self.n_layers * 8
        if physio.ndim not in [1, 2]:
            raise ValueError(f"Expected 1 or 2 dimensions, got {physio.ndim}")
        if physio.ndim == 1:
            physio = physio.unsqueeze(0)

        if physio.shape[-1] != expected_features:
            raise ValueError(
                f"Expected {expected_features} features, got {physio.shape[-1]}"
            )

        # get the number of samples
        device = physio.device
        dtype = physio.dtype

        # Move extinction coefficients to the same device
        eHbO2, eHb, wavelengths = self._cache_constants(dtype, device)

        # get the parameters
        sao2, vhb, a_mie, b_mie, a_ray, g, n, d = (
            physio[:, i + self.layer_base_ids] for i in range(8)
        )
        # transform to physical parameters
        physical = self._jit_transform_hb(
            sao2,
            vhb,
            a_mie,
            b_mie,
            a_ray,
            g,
            n,
            d,
            eHbO2,
            eHb,
            wavelengths,
            self.prefactor,
            self.eps,
            self.n_wavelengths,
            REFERENCE_WAVELENGTH_NM,
        )

        return physical

    @staticmethod
    @torch.jit.script  # type: ignore
    def _jit_transform_hb(
        sao2: torch.Tensor,
        vhb: torch.Tensor,
        a_mie: torch.Tensor,
        b_mie: torch.Tensor,
        a_ray: torch.Tensor,
        g: torch.Tensor,
        n: torch.Tensor,
        d: torch.Tensor,
        eHbO2: torch.Tensor,
        eHb: torch.Tensor,
        wavelengths: torch.Tensor,
        prefactor: float,
        eps: float,
        n_wavelengths: int,
        reference_wavelength: float,
        rayleigh_exponent: float = RAYLEIGH_EXPONENT,
    ) -> torch.Tensor:
        """JIT-compiled HB transformation."""

        # transform to physical parameters
        # mu_a = (sao2 * eHbO2 + (1 - sao2) * eHb) * vhb * cHb / gmw_hb
        mu_a = (
            sao2.unsqueeze(1) * eHbO2 + (1 - sao2).unsqueeze(1) * eHb
        ) * vhb.unsqueeze(1) * prefactor + eps

        # mu_s = a_mie * (wavelength / 500nm) ** (-b_mie) +
        # a_ray * (wavelength / 500nm) ** (-4)
        wavelength_factor = wavelengths / reference_wavelength

        if torch.all(a_ray == 0.0):
            mu_s = (
                a_mie.unsqueeze(1) * torch.pow(wavelength_factor, -b_mie.unsqueeze(1))
            ) / (1 - g.unsqueeze(1)) + eps
        else:
            mu_s = (
                a_mie.unsqueeze(1) * torch.pow(wavelength_factor, -b_mie.unsqueeze(1))
                + a_ray.unsqueeze(1) * torch.pow(wavelength_factor, -rayleigh_exponent)
            ) / (1 - g.unsqueeze(1)) + eps

        # concatenate all parameters, individual shapes:
        # (n_samples, n_wavelengths, n_layers)
        # order along n_layers axis: 3x mu_a, 3x mu_s, 3x g, 3x n, 3x d
        physical = torch.cat(
            [
                mu_a,
                mu_s,
                g.unsqueeze(1).expand(-1, n_wavelengths, -1),
                n.unsqueeze(1).expand(-1, n_wavelengths, -1),
                d.unsqueeze(1).expand(-1, n_wavelengths, -1),
            ],
            dim=-1,
        )

        return physical
