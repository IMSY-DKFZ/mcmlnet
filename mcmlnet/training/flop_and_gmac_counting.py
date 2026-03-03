"""
Define multiple methods to compare different FLOP and MAC counting tools.

Tools:
- PyTorch Lightning FLOPs counter: https://lightning.ai/docs/pytorch/stable/api_references.html#lightning.pytorch.utilities.measure_flops
- https://github.com/sovrasov/flops-counter.pytorch
- https://github.com/facebookresearch/fvcore/tree/main
- https://github.com/Lyken17/pytorch-OpCounter

Context:
- MAC & FLOP: https://github.com/sovrasov/flops-counter.pytorch/issues/16
- https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md

Additional installs:
```
pip install ptflops
pip install -U fvcore
pip install thop
```

"""

import copy

import lightning as pl
import torch
from fvcore.nn import FlopCountAnalysis
from lightning.pytorch.utilities import measure_flops
from ptflops import get_model_complexity_info
from thop import profile
from torch.utils.data import DataLoader

from mcmlnet.utils.logging import setup_logging

logger = setup_logging(level="info", logger_name=__name__)


class FLOPAndMACCounter:
    """Compare different FLOP and MAC counting tools for neural networks."""

    def __init__(self, model: pl.LightningModule) -> None:
        """Initialize counter with a PyTorch Lightning model.

        Args:
            model: PyTorch Lightning model to analyze

        Raises:
            TypeError: If model is not a PyTorch Lightning module
        """
        if not isinstance(model, pl.LightningModule):
            raise TypeError("Model must be a PyTorch Lightning module")
        self.model = model

    def _use_deep_copy(self) -> pl.LightningModule:
        """Use deep copy to avoid side effects.

        Returns:
            Deep copy of the model
        """
        return copy.deepcopy(self.model)

    def _to_giga_fmt(self, value: int | float, dec: int = 4) -> float:
        """Convert a value to common giga format (1e9)

        Args:
            value: Value to convert
            dec: Number of decimal places to round to

        Returns:
            Value in giga format
        """
        return round(value / 1e9, dec)

    def _check_model_params_warning(
        self, model_params: float | int | None, params: float | int
    ) -> None:
        """Check if model parameters match and raise warning if not.

        Args:
            model_params: Expected number of parameters
            params: Actual number of parameters from counting tool
        """
        if (
            model_params is not None and abs(params - model_params) > 1
        ):  # Allow small differences
            logger.warning(
                f"Model parameters mismatch: expected {model_params}, got {params}. "
                f"Difference: {abs(params - model_params)}"
            )

    def _model_loss(
        self, y: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Dummy loss function for backward pass computation.

        Args:
            y: Model output

        Returns:
            Scalar loss tensor
        """
        if isinstance(y, tuple):
            return y[0].sum()
        else:
            return y.sum()

    def _input(
        self, input_size: tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Generate dummy input tensor for model analysis.

        Args:
            input_size: Shape of input tensor (batch_size, ...)

        Returns:
            Tuple of (main_input, conditional_input)

        Raises:
            ValueError: If input size is invalid
        """
        if len(input_size) < 2:
            raise ValueError(
                f"Input size must have at least 2 dimensions, got {len(input_size)}"
            )

        # Check for conditional models
        ndim_c = getattr(self.model, "ndim_c", None)
        if ndim_c:
            conditional_input = torch.randn(input_size[0], ndim_c)
        else:
            conditional_input = None

        return torch.randn(input_size), conditional_input

    def pl_gflop_counter(self, input_size: tuple[int, ...]) -> float:
        """PyTorch Lightning GFLOPs counter for forward pass.

        Args:
            input_size: Input tensor shape

        Returns:
            GFLOPs for forward pass
        """
        model = self._use_deep_copy()
        with torch.device("meta"):
            model = model.to("meta")
            x, c = self._input(input_size)

        if c is not None:
            # supports conditional INNs
            def model_fwd() -> torch.Tensor:
                return model(x, c=[c])

        else:
            if hasattr(model, "ndim_c"):
                # supports unconditional INNs
                def model_fwd() -> torch.Tensor:
                    return model(x, c=[c])

            else:
                # supports other models
                def model_fwd() -> torch.Tensor:
                    return model(x)

        fwd_flops = measure_flops(model, model_fwd)
        return self._to_giga_fmt(fwd_flops)

    def pl_gflop_counter_fwd_and_bwd(self, input_size: tuple[int, ...]) -> float:
        """
        PyTorch Lightning GFLOPs counter for forward and backward pass
        """
        model = self._use_deep_copy()
        with torch.device("meta"):
            model = model.to("meta")
            x, c = self._input(input_size)

        if c is not None:

            def model_fwd() -> torch.Tensor:
                return model(x, c=[c])

        else:

            def model_fwd() -> torch.Tensor:
                return model(x)

        fwd_and_bwd_flops = measure_flops(model, model_fwd, self._model_loss)

        return self._to_giga_fmt(fwd_and_bwd_flops)

    def fvcore_gflop_counter(self, input_size: tuple[int, ...]) -> float:
        """
        FVCore GFLOPs counter
        """
        model = self._use_deep_copy().to("cpu")
        x, c = self._input(input_size)
        if c is not None:
            flops = FlopCountAnalysis(model, (x, [c]))
        else:
            flops = FlopCountAnalysis(model, x)
        # more detailed outputs:
        # flops.by_operator()
        # flops.by_module()
        # flops.by_module_and_operator()

        return self._to_giga_fmt(flops.total())

    def thop_gmac_counter(
        self, input_size: tuple[int, ...], model_params: float | int | None = None
    ) -> float:
        """
        THOP GMACs counter
        """
        model = self._use_deep_copy().to("cpu")
        x, c = self._input(input_size)
        if c is not None:
            macs, params = profile(model, inputs=(x, [c]), verbose=False)
        else:
            macs, params = profile(model, inputs=(x,), verbose=False)

        self._check_model_params_warning(model_params, params)

        return self._to_giga_fmt(macs)

    def ptflops_gmac_counter_pytorch(
        self, input_size: tuple[int, ...], model_params: float | int | None = None
    ) -> float:
        """PTFlops GMACs counter with PyTorch backend.

        Args:
            input_size: Input tensor shape
            model_params: Expected number of model parameters

        Returns:
            GMACs count
        """
        model = self._use_deep_copy().to("cpu")

        # Handle CUDA availability
        device_context = (
            torch.cuda.device(0) if torch.cuda.is_available() else torch.device("cpu")
        )

        with device_context:
            macs, params = get_model_complexity_info(
                model,
                input_size[1:],  # method automatically adds batch dimension
                as_strings=False,
                backend="pytorch",
                print_per_layer_stat=False,
                verbose=False,
            )
        self._check_model_params_warning(model_params, params)
        return self._to_giga_fmt(macs)

    def ptflops_gmac_counter_aten(
        self, input_size: tuple[int, ...], model_params: float | int | None = None
    ) -> float:
        """
        PTFlops GMACs counter
        """
        model = self._use_deep_copy().to("cpu")
        with torch.cuda.device(0):
            macs, params = get_model_complexity_info(
                model,
                input_size[1:],  # method automatically adds batch dimension
                as_strings=False,
                backend="aten",
                print_per_layer_stat=False,
                verbose=False,
            )
        self._check_model_params_warning(model_params, params)

        return self._to_giga_fmt(macs)

    def __call__(
        self,
        train_loader: DataLoader,
        n_trainable_params: int | None = None,
        per_epoch: bool = True,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Compute GFLOPs and GMACs and return results from different tools.

        Args:
            train_loader: DataLoader instance of the training set
            n_trainable_params: Number of trainable parameters in the model
            per_epoch: Whether to compute GFLOPs and GMACs per epoch (or per batch)

        Returns:
            Tuple of dictionaries containing GFLOP and GMAC results

        Raises:
            ValueError: If train_loader is empty or invalid
        """
        if len(train_loader) == 0:
            raise ValueError("DataLoader is empty")

        _example_batch = next(iter(train_loader))
        n_train_batch = len(train_loader)

        # Determine input size based on model type
        if getattr(self.model, "ndim_c", None) is not None:
            # supports conditional INNs
            input_size = tuple(_example_batch[1].shape)
        else:
            # supports regular MLPs, KANs, and unconditional INNs
            if isinstance(_example_batch, list | tuple) and len(_example_batch) >= 2:
                input_size = tuple(_example_batch[0].shape)
            else:
                input_size = tuple(_example_batch.shape)

        # GFLOPs computation
        pl_fwd = self.pl_gflop_counter(input_size)
        pl_fwd_bwd = self.pl_gflop_counter_fwd_and_bwd(input_size)
        fvcore = self.fvcore_gflop_counter(input_size)
        gflop_summary = {
            "GFLOP PL FWD": pl_fwd,
            "GFLOP PL FWD+BWD": pl_fwd_bwd,
            "GFLOP FVCore": fvcore,
        }
        # GMACs
        thop = self.thop_gmac_counter(input_size, n_trainable_params)
        # does not support conditional models
        ptflops_pt, ptflops_aten = (
            (0, 0)
            if getattr(self.model, "ndim_c", None)
            else (
                self.ptflops_gmac_counter_pytorch(input_size, n_trainable_params),
                self.ptflops_gmac_counter_aten(input_size, n_trainable_params),
            )
        )
        gmac_summary = {
            "GMAC THOP": thop,
            "GMAC PTFlops PyTorch": ptflops_pt,
            "GMAC PTFlops Aten": ptflops_aten,
        }
        # compute GFLOPs and GMACs per epoch
        # by multiplying with number of training batches
        if per_epoch:
            gflop_summary = {
                key: value * n_train_batch for key, value in gflop_summary.items()
            }
            gmac_summary = {
                key: value * n_train_batch for key, value in gmac_summary.items()
            }

        return gflop_summary, gmac_summary
