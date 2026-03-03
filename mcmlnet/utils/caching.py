"""Caching utilities."""

import os
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv

from mcmlnet.utils.logging import setup_logging

load_dotenv()
logger = setup_logging(level="info", logger_name=__name__)


def torch_to_np(
    data: torch.Tensor | np.ndarray | tuple[Any] | list[Any],
) -> torch.Tensor | np.ndarray | tuple[torch.Tensor, ...] | None:
    """Convert PyTorch tensors safely to NumPy arrays.

    Args:
        data: Function is looking for a tensor or a tuple tensors.

    Returns:
        Array or a tuple of arrays.
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, tuple | list):
        converted_items = []
        for i, item in enumerate(data):
            if isinstance(item, torch.Tensor | np.ndarray | tuple | list):
                converted_items.append(torch_to_np(item))
            else:
                logger.warning(f"Skipping non-tensor item at index {i}: {type(item)}")  # type: ignore [unreachable]
        return tuple(converted_items) if converted_items else None
    else:
        return None


def np_cache_to_file(func: Callable, file_name: str) -> Callable:
    """Cache numpy arrays to file.

    Args:
        func: Function to wrap around. Needs to return np.ndarray
            or tuple of np.ndarrays.
        file_name: Name of the file or abspath to save to.

    Returns:
        Wrapped function with caching.

    Raises:
        TypeError: If the cache file does not contain a valid numpy array or tuple.
    """
    if not os.path.isabs(file_name):
        file = os.path.join(os.environ["cache_dir"], file_name)
        os.makedirs(os.path.dirname(file), exist_ok=True)
    else:
        file = file_name

    def wrapper(*args: Any, **kwargs: Any) -> np.ndarray | tuple[np.ndarray, ...]:
        # Check for existence of cache file
        try:
            # Load cache from file
            cache = np.load(file, allow_pickle=True)
            logger.info(f"Loading results from {file}")
            if isinstance(cache, np.lib.npyio.NpzFile):
                # Unpack dict-like, single entry needs special unpacking
                if len(cache.values()) == 1:
                    cache = cache.f.arr_0
                else:
                    cache = tuple(cache.values())
            else:
                raise TypeError(
                    f"Cache file {file} does not contain a valid numpy array or tuple."
                )
        except OSError:
            # If file does not exist, call the function and save the result
            logger.info(f"Saving results to {file}")
            cache = func(*args, **kwargs)
            if isinstance(cache, tuple | list):
                # NOTE: Saving lists as tuples -> returns tuples on >2nd call
                np.savez_compressed(file, *torch_to_np(cache))
            else:
                np.savez_compressed(file, torch_to_np(cache))
        return cache

    return wrapper
