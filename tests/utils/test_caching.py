import copy
import os
from pathlib import Path

import numpy as np
import torch

from mcmlnet.utils.caching import (
    np_cache_to_file,
    torch_to_np,
)


class TestTorchToNp:
    def test_torch_to_np(self) -> None:
        """Test conversion from torch tensors to numpy arrays."""
        t = torch.tensor([1.0, 2.0])
        arr = torch_to_np(t)
        assert isinstance(arr, np.ndarray)

    def test_torch_to_np_no_conversion(self) -> None:
        """Test that non-tensor input returns None or unchanged."""
        arr = np.array([1.0, 2.0])
        converted = torch_to_np(copy.deepcopy(arr))
        assert np.array_equal(converted, arr)

    def test_torch_to_np_tuple(self) -> None:
        """Test conversion of tuple of tensors."""
        tup = (torch.tensor([1.0]), torch.tensor([2.0]))
        arr_tup = torch_to_np(tup)
        assert isinstance(arr_tup, tuple)
        assert isinstance(arr_tup[0], np.ndarray)
        assert isinstance(arr_tup[1], np.ndarray)

    def test_torch_to_np_list(self) -> None:
        """Test conversion of list of tensors."""
        lst = [torch.tensor([1.0]), torch.tensor([2.0])]
        arr_lst = torch_to_np(lst)
        assert isinstance(arr_lst, tuple)
        assert isinstance(arr_lst[0], np.ndarray)
        assert isinstance(arr_lst[1], np.ndarray)

    def test_torch_to_np_mixed_tuple(self) -> None:
        """Test conversion of mixed tuple."""
        tup = (torch.tensor([1.0]), np.array([2.0]), 42)
        arr_tup = torch_to_np(tup)
        assert isinstance(arr_tup, tuple)
        assert isinstance(arr_tup[0], np.ndarray)
        assert arr_tup[1] is tup[1]
        assert len(arr_tup) == 2  # Non-tensor item is skipped

    def test_torch_to_np_mixed_list(self) -> None:
        """Test conversion of mixed list."""
        lst = [torch.tensor([1.0]), np.array([2.0]), 42]
        arr_lst = torch_to_np(lst)
        assert isinstance(arr_lst, tuple)
        assert isinstance(arr_lst[0], np.ndarray)
        assert arr_lst[1] is lst[1]
        assert len(arr_lst) == 2  # Non-tensor item is skipped

    def test_torch_to_np_nested_tuple(self) -> None:
        """Test conversion of nested tuple."""
        tup = (torch.tensor([1.0]), (torch.tensor([2.0]), 42))
        arr_tup = torch_to_np(tup)
        assert isinstance(arr_tup, tuple)
        assert isinstance(arr_tup[0], np.ndarray)
        assert isinstance(arr_tup[1], tuple)
        assert isinstance(arr_tup[1][0], np.ndarray)
        assert len(arr_tup) == 2
        assert len(arr_tup[1]) == 1  # Non-tensor item is skipped

    def test_torch_to_np_nested_list(self) -> None:
        """Test conversion of nested list."""
        lst = [torch.tensor([1.0]), [torch.tensor([2.0]), 42]]
        arr_lst = torch_to_np(lst)
        assert isinstance(arr_lst, tuple)
        assert isinstance(arr_lst[0], np.ndarray)
        assert isinstance(arr_lst[1], tuple)
        assert isinstance(arr_lst[1][0], np.ndarray)
        assert len(arr_lst) == 2
        assert len(arr_lst[1]) == 1  # Non-tensor item is skipped

    def test_torch_to_np_any_inputs(self) -> None:
        """Test conversion of other types to None (pass without error)."""
        assert torch_to_np(()) is None
        assert torch_to_np([]) is None
        assert torch_to_np(42) is None
        assert torch_to_np(42.0) is None
        assert torch_to_np("string") is None
        assert torch_to_np(None) is None


class TestCachingDecorators:
    def setup_method(self) -> None:
        """Setup common variables."""
        self.arr = np.arange(5)
        self.tensor = torch.tensor(self.arr)

    def test_np_cache_to_file(self, tmp_path: Path) -> None:
        """Test numpy caching decorator."""

        def func() -> np.ndarray:
            return copy.deepcopy(self.arr)

        file = tmp_path / "test.npz"
        cached_func = np_cache_to_file(func, str(file))
        # First call saves
        out1 = cached_func()
        assert os.path.exists(file)
        assert isinstance(out1, np.ndarray)
        assert np.array_equal(out1, self.arr)
        # Second call loads
        out2 = cached_func()
        assert isinstance(out2, np.ndarray)
        assert np.array_equal(out2, self.arr)

    def test_np_cache_to_file_tuple(self, tmp_path: Path) -> None:
        """Test numpy caching decorator with tuple return."""

        def func() -> tuple[np.ndarray, np.ndarray]:
            return (copy.deepcopy(self.arr), copy.deepcopy(self.arr) + 1)

        file = tmp_path / "test2.npz"
        cached_func = np_cache_to_file(func, str(file))
        # First call saves
        out1 = cached_func()
        assert os.path.exists(file)
        assert isinstance(out1, tuple)
        assert np.array_equal(out1[0], self.arr)
        assert np.array_equal(out1[1], self.arr + 1)
        # Second call loads
        out2 = cached_func()
        assert isinstance(out2, tuple)
        assert np.array_equal(out2[0], self.arr)
        assert np.array_equal(out2[1], self.arr + 1)

    def test_np_cache_to_file_tuple_of_tensors(self, tmp_path: Path) -> None:
        """Test numpy caching decorator with tuple of tensors."""

        def func() -> tuple[torch.Tensor, torch.Tensor]:
            tensor = copy.deepcopy(self.tensor)
            return (tensor, tensor + 1)

        file = tmp_path / "test3.npz"
        cached_func = np_cache_to_file(func, str(file))
        # First call saves
        out1 = cached_func()
        assert os.path.exists(file)
        assert isinstance(out1, tuple)
        assert np.array_equal(out1[0], self.arr)
        assert np.array_equal(out1[1], self.arr + 1)
        # Second call loads
        out2 = cached_func()
        assert isinstance(out2, tuple)
        assert np.array_equal(out2[0], self.arr)
        assert np.array_equal(out2[1], self.arr + 1)

    def test_np_cache_to_file_list(self, tmp_path: Path) -> None:
        """Test numpy caching decorator with list return."""

        def func() -> list[np.ndarray]:
            return [copy.deepcopy(self.arr), copy.deepcopy(self.arr) + 1]

        file = tmp_path / "test4.npz"
        cached_func = np_cache_to_file(func, str(file))
        # First call saves
        out1 = cached_func()
        assert os.path.exists(file)
        assert isinstance(out1, list)
        assert np.array_equal(out1[0], self.arr)
        assert np.array_equal(out1[1], self.arr + 1)
        # Second call loads
        out2 = cached_func()
        assert isinstance(out2, tuple)
        assert np.array_equal(out2[0], self.arr)
        assert np.array_equal(out2[1], self.arr + 1)

    def test_np_cache_to_file_list_of_tensors(self, tmp_path: Path) -> None:
        """Test numpy caching decorator with list of tensors."""

        def func() -> list[torch.Tensor]:
            tensor = copy.deepcopy(self.tensor)
            return [tensor, tensor + 1]

        file = tmp_path / "test5.npz"
        cached_func = np_cache_to_file(func, str(file))
        # First call saves
        out1 = cached_func()
        assert os.path.exists(file)
        assert isinstance(out1, list)
        assert np.array_equal(out1[0], self.arr)
        assert np.array_equal(out1[1], self.arr + 1)
        # Second call loads
        out2 = cached_func()
        assert isinstance(out2, tuple)
        assert np.array_equal(out2[0], self.arr)
        assert np.array_equal(out2[1], self.arr + 1)
