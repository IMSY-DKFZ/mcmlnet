"""Tensor utilities and type conversions."""

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeAlias

import numpy as np
import torch

TensorType: TypeAlias = torch.Tensor | np.ndarray


def convert_to_torch(
    data: TensorType,
) -> tuple[torch.Tensor, bool, torch.device | None]:
    """Convert data to torch tensor, tracking if it was originally numpy.

    Args:
        data: Input data.

    Returns:
        Tuple of (converted tensor, was_numpy flag, device).

    Raises:
        ValueError: If the input numpy array contains non-finite values (NaN or Inf).
        TypeError: If the input data is not a numpy array or torch tensor.
    """
    if isinstance(data, np.ndarray):
        if not np.isfinite(data).all():
            raise ValueError(
                "Input numpy array contains non-finite values (NaN or Inf)"
            )
        return torch.from_numpy(data), True, None
    elif isinstance(data, torch.Tensor):
        return data, False, data.device
    else:
        raise TypeError(
            f"Unsupported type {type(data)}! Supported types: np.ndarray, torch.Tensor."
        )


def convert_to_numpy(data: TensorType) -> tuple[np.ndarray, bool, torch.device | None]:
    """Convert data to numpy array, tracking if it was originally torch tensor.

    Args:
        data: Input data.

    Returns:
        Tuple of (converted array, was_torch flag, device).

    Raises:
        ValueError: If the input torch tensor contains non-finite values (NaN or Inf).
        TypeError: If the input data is not a numpy array or torch tensor.
    """
    if isinstance(data, np.ndarray):
        return data, False, None
    elif isinstance(data, torch.Tensor):
        if not torch.isfinite(data).all():
            raise ValueError(
                "Input torch tensor contains non-finite values (NaN or Inf)"
            )
        device = data.device
        return data.detach().cpu().numpy(), True, device
    else:
        raise TypeError(
            f"Unknown type {type(data)}! "
            "Possible types are torch.Tensor and np.ndarray."
        )


def convert_output_to_input_type(
    result: TensorType, was_numpy: bool, device: torch.device | None
) -> TensorType:
    """Convert result back to original type.

    Args:
        result: The result to convert.
        was_numpy: Whether the original data was numpy.
        device: The device the torch.Tensor was on.

    Returns:
        The converted result.
    """
    if was_numpy and isinstance(result, torch.Tensor):
        return result.detach().cpu().numpy()
    elif not was_numpy and isinstance(result, np.ndarray):
        tensor = torch.from_numpy(result)
        if device is not None:
            tensor = tensor.to(device)
        return tensor
    else:
        return result


def tensor_conversion_decorator(func: Callable) -> Callable:
    """Automatically convert input data to torch tensor and convert the output
    back to numpy array if the input data was a numpy array.

    Args:
        func: The function to decorate.

    Returns:
        The decorated function.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Convert all tensor/array arguments to torch
        converted_args = []
        conversion_info = []  # Store (was_numpy, device) for each converted arg

        for arg in args:
            if isinstance(arg, torch.Tensor | np.ndarray):
                converted_arg, was_numpy, device = convert_to_torch(arg)
                converted_args.append(converted_arg)
                conversion_info.append((was_numpy, device))
            else:
                converted_args.append(arg)
                conversion_info.append((None, None))  # type: ignore [arg-type]

        # Convert tensor/array keyword arguments to torch
        converted_kwargs = {}
        kwargs_conversion_info = {}

        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor | np.ndarray):
                converted_value, was_numpy, device = convert_to_torch(value)
                converted_kwargs[key] = converted_value
                kwargs_conversion_info[key] = (was_numpy, device)
            else:
                converted_kwargs[key] = value
                kwargs_conversion_info[key] = (None, None)  # type: ignore [assignment]

        # Call the function with converted arguments
        result = func(*converted_args, **converted_kwargs)

        if result is None:
            return None

        # Helper function to convert back using original type info
        def convert_back(
            value: Any, original_was_numpy: bool, original_device: torch.device | None
        ) -> Any:
            if isinstance(value, torch.Tensor | np.ndarray):
                return convert_output_to_input_type(
                    value, original_was_numpy, original_device
                )
            return value

        # Determine the primary conversion info (from first tensor argument)
        primary_was_numpy, primary_device = None, None
        for info in conversion_info:
            primary_was_numpy, primary_device = info
            if primary_was_numpy is not None:
                # Found the first tensor argument, use its conversion info
                break

        # If no tensor args, check kwargs
        if primary_was_numpy is None:
            for info in kwargs_conversion_info.values():
                primary_was_numpy, primary_device = info
                if primary_was_numpy is not None:
                    # Found the first tensor argument in kwargs, use its conversion info
                    break

        # Convert output back using primary conversion info
        if primary_was_numpy is not None:
            if isinstance(result, tuple):
                return tuple(
                    convert_back(r, primary_was_numpy, primary_device) for r in result
                )
            elif isinstance(result, list):
                return [
                    convert_back(r, primary_was_numpy, primary_device) for r in result
                ]
            else:
                return convert_back(result, primary_was_numpy, primary_device)

        return result

    return wrapper


def array_conversion_decorator(func: Callable) -> Callable:
    """Automatically convert input data to numpy array and convert the output
    back to torch tensor if the input data was a torch tensor.

    Args:
        func: The function to decorate.

    Returns:
        The decorated function.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Convert all tensor/array arguments to numpy
        converted_args = []
        conversion_info = []  # Store (was_torch, device) for each converted arg

        for arg in args:
            if isinstance(arg, torch.Tensor | np.ndarray):
                converted_arg, was_torch, device = convert_to_numpy(arg)
                converted_args.append(converted_arg)
                conversion_info.append((was_torch, device))
            else:
                converted_args.append(arg)
                conversion_info.append((None, None))  # type: ignore [arg-type]

        # Convert tensor/array keyword arguments to numpy
        converted_kwargs = {}
        kwargs_conversion_info = {}

        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor | np.ndarray):
                converted_value, was_torch, device = convert_to_numpy(value)
                converted_kwargs[key] = converted_value
                kwargs_conversion_info[key] = (was_torch, device)
            else:
                converted_kwargs[key] = value
                kwargs_conversion_info[key] = (None, None)  # type: ignore [assignment]

        # Call the function with converted arguments
        result = func(*converted_args, **converted_kwargs)

        if result is None:
            return None

        # Helper function to convert back using original type info
        def convert_back(
            value: Any, original_was_torch: bool, original_device: torch.device | None
        ) -> Any:
            if isinstance(value, torch.Tensor | np.ndarray):
                return convert_output_to_input_type(
                    value, not original_was_torch, original_device
                )
            return value

        # Determine the primary conversion info (from first tensor argument)
        primary_was_torch, primary_device = None, None
        for info in conversion_info:
            primary_was_torch, primary_device = info
            if primary_was_torch is not None:
                # Found the first tensor argument, use its conversion info
                break

        # If no tensor args, check kwargs
        if primary_was_torch is None:
            for info in kwargs_conversion_info.values():
                primary_was_torch, primary_device = info
                if primary_was_torch is not None:
                    # Found the first tensor argument in kwargs, use its conversion info
                    break

        # Convert output back using primary conversion info
        if primary_was_torch is not None:
            if isinstance(result, tuple):
                return tuple(
                    convert_back(r, primary_was_torch, primary_device) for r in result
                )
            elif isinstance(result, list):
                return [
                    convert_back(r, primary_was_torch, primary_device) for r in result
                ]
            else:
                return convert_back(result, primary_was_torch, primary_device)

        return result

    return wrapper
