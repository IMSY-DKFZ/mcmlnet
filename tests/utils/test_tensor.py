"""Tests for mcmlnet.utils.tensor module."""

import numpy as np
import pytest
import torch

from mcmlnet.utils.tensor import (
    array_conversion_decorator,
    convert_to_numpy,
    convert_to_torch,
    tensor_conversion_decorator,
)


class TestBasicMethods:
    """Test cases for basic conversion functions."""

    def test_convert_to_tensor_raises_type_error(self) -> None:
        """Test convert_to_tensor raises TypeError for unsupported types."""
        with pytest.raises(TypeError):
            convert_to_torch("invalid_type_str")

    def test_convert_to_numpy_raises_type_error(self) -> None:
        """Test convert_to_numpy raises TypeError for unsupported types."""
        with pytest.raises(TypeError):
            convert_to_numpy("invalid_type_str")


class TestTensorUtils:
    """Test cases for tensor utility functions."""

    def setup_method(self) -> None:
        """Setup common test data."""
        self.numpy_input = np.array([1.0, 2.0, 3.0])
        self.torch_input = torch.tensor([1.0, 2.0, 3.0])
        self.optional_input = torch.tensor([0.5, 0.5, 0.5])

    @pytest.mark.parametrize(  # type: ignore[misc]
        "input_data,expected_type,expected_result",
        [
            ("torch_input", torch.Tensor, torch.tensor([2.0, 4.0, 6.0])),
            ("numpy_input", np.ndarray, np.array([2.0, 4.0, 6.0])),
        ],
    )
    def test_tensor_conversion_decorator(
        self,
        input_data: str,
        expected_type: type,
        expected_result: torch.Tensor | np.ndarray,
    ) -> None:
        """Test tensor_conversion_decorator with single tensor argument."""

        @tensor_conversion_decorator
        def test_function(data: torch.Tensor | np.ndarray) -> torch.Tensor:
            if not isinstance(data, torch.Tensor):
                raise ValueError("Input must be a torch tensor")
            return data * 2

        result = test_function(getattr(self, input_data))
        assert isinstance(result, expected_type)
        if isinstance(result, np.ndarray):
            assert np.allclose(result, expected_result)
        else:
            assert torch.equal(result, expected_result)

    @pytest.mark.parametrize(  # type: ignore[misc]
        "input_data",
        [
            np.array([np.inf, 2.0, 3.0]),
            np.array([np.nan, 2.0, 3.0]),
        ],
    )
    def test_tensor_conversion_decorator_with_non_finite_values(
        self, input_data: np.ndarray
    ) -> None:
        """Test tensor_conversion_decorator raises ValueError for non-finite values."""

        @tensor_conversion_decorator
        def test_function(data: torch.Tensor | np.ndarray) -> torch.Tensor:
            if not isinstance(data, torch.Tensor):
                raise ValueError("Input must be a torch tensor")
            return data

        with pytest.raises(ValueError):
            test_function(input_data)

    @pytest.mark.parametrize(  # type: ignore[misc]
        "a1,a2,expected_type,expected_result",
        [
            ("torch_input", "numpy_input", torch.Tensor, torch.tensor([2.0, 4.0, 6.0])),
            ("numpy_input", "torch_input", np.ndarray, np.array([2.0, 4.0, 6.0])),
        ],
    )
    def test_tensor_conversion_decorator_with_multiple_args(
        self,
        a1: torch.Tensor | np.ndarray,
        a2: torch.Tensor | np.ndarray,
        expected_type: type,
        expected_result: torch.Tensor | np.ndarray,
    ) -> None:
        """Test tensor_conversion_decorator with multiple tensor arguments."""

        @tensor_conversion_decorator
        def test_function(
            tensor1: torch.Tensor | np.ndarray, tensor2: torch.Tensor | np.ndarray
        ) -> torch.Tensor:
            if not isinstance(tensor1, torch.Tensor):
                raise ValueError("Input must be a torch tensor")
            if not isinstance(tensor2, torch.Tensor):
                raise ValueError("Input must be a torch tensor")
            return tensor1 + tensor2

        result = test_function(getattr(self, a1), getattr(self, a2))
        assert isinstance(result, expected_type)
        if isinstance(result, np.ndarray):
            assert np.allclose(result, expected_result)
        else:
            assert torch.equal(result, expected_result)

    def test_tensor_conversion_decorator_with_non_tensor_args(self) -> None:
        """Test tensor_conversion_decorator with non-tensor arguments."""

        @tensor_conversion_decorator
        def test_function(tensor: torch.Tensor, multiplier: float) -> torch.Tensor:
            if not isinstance(tensor, torch.Tensor):
                raise ValueError("Input must be a torch tensor")
            return tensor * multiplier

        result = test_function(self.torch_input, 2.5)
        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, torch.tensor([2.5, 5.0, 7.5]))

    def test_tensor_conversion_decorator_with_kwargs(self) -> None:
        """Test tensor_conversion_decorator with keyword arguments."""

        @tensor_conversion_decorator
        def test_function(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
            if not isinstance(tensor, torch.Tensor):
                raise ValueError("Input must be a torch tensor")
            return tensor.mean(dim=dim)

        result = test_function(torch.randn(5, 10), dim=1)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (5,)

    def test_tensor_conversion_decorator_return_type(self) -> None:
        """Test tensor_conversion_decorator preserves return type."""

        @tensor_conversion_decorator
        def test_function(tensor: torch.Tensor) -> float:
            if not isinstance(tensor, torch.Tensor):
                raise ValueError("Input must be a torch tensor")
            return float(tensor.sum())

        result = test_function(self.torch_input)
        assert isinstance(result, float)
        assert result == 6.0

    @pytest.mark.parametrize(  # type: ignore[misc]
        "input_data,expected_shape,expected_type",
        [
            (torch.randn(10, 5), (5,), torch.Tensor),
            (torch.randn(10, 5, 3), (5, 3), torch.Tensor),
            (np.random.randn(10, 5), (5,), np.ndarray),
            (np.random.randn(10, 5, 3), (5, 3), np.ndarray),
        ],
    )
    def test_tensor_conversion_decorator_with_complex_shapes(
        self,
        input_data: torch.Tensor | np.ndarray,
        expected_shape: tuple[int, ...],
        expected_type: torch.Tensor | np.ndarray,
    ) -> None:
        """Test tensor_conversion_decorator with complex tensor shapes."""

        @tensor_conversion_decorator
        def test_function(tensor: torch.Tensor | np.ndarray) -> torch.Tensor:
            return (
                tensor.mean(dim=0)
                if isinstance(tensor, torch.Tensor)
                else tensor.mean(axis=0)
            )

        result = test_function(input_data)
        assert isinstance(result, expected_type)
        assert result.shape == expected_shape

    def test_tensor_conversion_decorator_error_handling(self) -> None:
        """Test tensor_conversion_decorator error handling."""

        @tensor_conversion_decorator
        def test_function(tensor: torch.Tensor) -> torch.Tensor:
            if not isinstance(tensor, torch.Tensor):
                raise ValueError("Input must be a torch tensor")
            raise RuntimeError("Test error")

        with pytest.raises(RuntimeError):
            test_function(self.torch_input)

    def test_tensor_conversion_decorator_device_handling(self) -> None:
        """Test tensor_conversion_decorator preserves device of torch tensors."""

        @tensor_conversion_decorator
        def test_function(data: torch.Tensor | np.ndarray) -> torch.Tensor:
            if not isinstance(data, torch.Tensor):
                raise ValueError("Input must be a torch tensor")
            return data * 2

        result = test_function(self.torch_input.clone())
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cpu"
        if torch.cuda.is_available():
            result = test_function(self.torch_input.clone().to("cuda"))
            assert isinstance(result, torch.Tensor)
            assert result.device.type == "cuda"

    @pytest.mark.parametrize(  # type: ignore[misc]
        "input_data,expected_type",
        [
            ("torch_input", torch.Tensor),
            ("numpy_input", np.ndarray),
        ],
    )
    def test_tensor_conversion_decorator_with_tuples(
        self, input_data: str, expected_type: type
    ) -> None:
        """Test tensor_conversion_decorator with tuple return types."""

        @tensor_conversion_decorator
        def test_function(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if not isinstance(tensor, torch.Tensor):
                raise ValueError("Input must be a torch tensor")
            return tensor * 2, tensor * 3

        result1, result2 = test_function(getattr(self, input_data))
        assert isinstance(result1, expected_type)
        assert isinstance(result2, expected_type)
        if isinstance(result1, np.ndarray):
            assert np.allclose(result1, np.array([2.0, 4.0, 6.0]))
            assert np.allclose(result2, np.array([3.0, 6.0, 9.0]))
        else:
            assert torch.equal(result1, torch.tensor([2.0, 4.0, 6.0]))
            assert torch.equal(result2, torch.tensor([3.0, 6.0, 9.0]))

    @pytest.mark.parametrize(  # type: ignore[misc]
        "input_data,expected_type",
        [
            ("torch_input", torch.Tensor),
            ("numpy_input", np.ndarray),
        ],
    )
    def test_tensor_conversion_decorator_with_lists(
        self, input_data: str, expected_type: type
    ) -> None:
        """Test tensor_conversion_decorator with list return types."""

        @tensor_conversion_decorator
        def test_function(tensor: torch.Tensor) -> list[torch.Tensor]:
            if not isinstance(tensor, torch.Tensor):
                raise ValueError("Input must be a torch tensor")
            return [tensor * 2, tensor * 3]

        results = test_function(getattr(self, input_data))
        assert isinstance(results, list)
        assert len(results) == 2
        assert isinstance(results[0], expected_type)
        assert isinstance(results[1], expected_type)
        if isinstance(results[0], np.ndarray):
            assert np.allclose(results[0], np.array([2.0, 4.0, 6.0]))
            assert np.allclose(results[1], np.array([3.0, 6.0, 9.0]))
        else:
            assert torch.equal(results[0], torch.tensor([2.0, 4.0, 6.0]))
            assert torch.equal(results[1], torch.tensor([3.0, 6.0, 9.0]))

    def test_tensor_conversion_decorator_no_tensor_args(self) -> None:
        """Test tensor_conversion_decorator with no tensor arguments."""

        @tensor_conversion_decorator
        def test_function(x: float, y: float) -> float:
            return x + y

        result = test_function(1.0, 2.0)
        assert isinstance(result, float)
        assert result == 3.0

    def test_tensor_conversion_decorator_none_result(self) -> None:
        """Test tensor_conversion_decorator with None result."""

        @tensor_conversion_decorator
        def test_function(tensor: torch.Tensor) -> None:
            if not isinstance(tensor, torch.Tensor):
                raise ValueError("Input must be a torch tensor")
            return None

        result = test_function(self.torch_input)
        assert result is None


class TestArrayUtils:
    """Test cases for array conversion decorator."""

    def setup_method(self) -> None:
        self.numpy_input = np.array([1.0, 2.0, 3.0])
        self.torch_input = torch.tensor([1.0, 2.0, 3.0])
        self.optional_input = np.array([0.5, 0.5, 0.5])

    @pytest.mark.parametrize(  # type: ignore[misc]
        "input_data,expected_type,expected_result",
        [
            ("numpy_input", np.ndarray, np.array([2.0, 4.0, 6.0])),
            ("torch_input", torch.Tensor, torch.tensor([2.0, 4.0, 6.0])),
        ],
    )
    def test_array_conversion_decorator(
        self,
        input_data: str,
        expected_type: type,
        expected_result: torch.Tensor | np.ndarray,
    ) -> None:
        """Test array_conversion_decorator with single array argument."""

        @array_conversion_decorator
        def test_function(array: np.ndarray) -> np.ndarray:
            if not isinstance(array, np.ndarray):
                raise ValueError("Input must be a numpy array")
            return array * 2

        result = test_function(getattr(self, input_data))
        assert isinstance(result, expected_type)
        if isinstance(result, np.ndarray):
            assert np.allclose(result, expected_result)
        else:
            assert torch.equal(result, expected_result)

    @pytest.mark.parametrize(  # type: ignore[misc]
        "input_data",
        [
            torch.tensor([torch.inf, 2.0, 3.0]),
            torch.tensor([torch.nan, 2.0, 3.0]),
        ],
    )
    def test_array_conversion_decorator_with_non_finite_values(
        self, input_data: torch.Tensor
    ) -> None:
        """Test array_conversion_decorator raises ValueError for non-finite values."""

        @array_conversion_decorator
        def test_function(array: np.ndarray) -> np.ndarray:
            if not isinstance(array, np.ndarray):
                raise ValueError("Input must be a numpy array")
            return array

        with pytest.raises(ValueError):
            test_function(input_data)

    @pytest.mark.parametrize(  # type: ignore[misc]
        "a1,a2,expected_type,expected_result",
        [
            ("numpy_input", "torch_input", np.ndarray, np.array([2.0, 4.0, 6.0])),
            ("torch_input", "numpy_input", torch.Tensor, torch.tensor([2.0, 4.0, 6.0])),
        ],
    )
    def test_array_conversion_decorator_with_multiple_args(
        self,
        a1: str,
        a2: str,
        expected_type: type,
        expected_result: torch.Tensor | np.ndarray,
    ) -> None:
        """Test array_conversion_decorator with multiple array arguments."""

        @array_conversion_decorator
        def test_function(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
            if not isinstance(array1, np.ndarray):
                raise ValueError("Input must be a numpy array")
            if not isinstance(array2, np.ndarray):
                raise ValueError("Input must be a numpy array")
            return array1 + array2

        result = test_function(getattr(self, a1), getattr(self, a2))
        assert isinstance(result, expected_type)
        if isinstance(result, np.ndarray):
            assert np.allclose(result, expected_result)
        else:
            assert torch.equal(result, expected_result)

    def test_array_conversion_decorator_with_non_array_args(self) -> None:
        """Test array_conversion_decorator with non-array arguments."""

        @array_conversion_decorator
        def test_function(array: np.ndarray, multiplier: float) -> np.ndarray:
            if not isinstance(array, np.ndarray):
                raise ValueError("Input must be a numpy array")
            return array * multiplier

        result = test_function(self.numpy_input, 2.5)
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, np.array([2.5, 5.0, 7.5]))

    def test_array_conversion_decorator_with_kwargs(self) -> None:
        """Test array_conversion_decorator with keyword arguments."""

        @array_conversion_decorator
        def test_function(array: np.ndarray, axis: int = 0) -> np.ndarray:
            if not isinstance(array, np.ndarray):
                raise ValueError("Input must be a numpy array")
            return array.mean(axis=axis)

        numpy_input = np.random.randn(5, 10)
        result = test_function(numpy_input, axis=1)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)

    def test_array_conversion_decorator_return_type(self) -> None:
        """Test array_conversion_decorator preserves return type."""

        @array_conversion_decorator
        def test_function(array: np.ndarray) -> float:
            if not isinstance(array, np.ndarray):
                raise ValueError("Input must be a numpy array")
            return float(array.sum())

        result = test_function(self.numpy_input)
        assert isinstance(result, float)
        assert result == 6.0

    @pytest.mark.parametrize(  # type: ignore[misc]
        "input_data,expected_shape,expected_type",
        [
            (np.random.randn(10, 5), (5,), np.ndarray),
            (np.random.randn(10, 5, 3), (5, 3), np.ndarray),
            (torch.randn(10, 5), (5,), torch.Tensor),
            (torch.randn(10, 5, 3), (5, 3), torch.Tensor),
        ],
    )
    def test_array_conversion_decorator_with_complex_shapes(
        self,
        input_data: torch.Tensor | np.ndarray,
        expected_shape: tuple[int, ...],
        expected_type: type,
    ) -> None:
        """Test array_conversion_decorator with complex array shapes."""

        @array_conversion_decorator
        def test_function(array: np.ndarray) -> np.ndarray:
            if not isinstance(array, np.ndarray):
                raise ValueError("Input must be a numpy array")
            return array.mean(axis=0)

        result = test_function(input_data)
        assert isinstance(result, expected_type)
        assert result.shape == expected_shape  # type: ignore[attr-defined]

    def test_array_conversion_decorator_error_handling(self) -> None:
        """Test array_conversion_decorator error handling."""

        @array_conversion_decorator
        def test_function(array: np.ndarray) -> np.ndarray:
            if not isinstance(array, np.ndarray):
                raise ValueError("Input must be a numpy array")
            raise RuntimeError("Test error")

        with pytest.raises(RuntimeError):
            test_function(self.numpy_input)

    def test_array_conversion_decorator_device_handling(self) -> None:
        """Test array_conversion_decorator with GPU tensors."""

        @array_conversion_decorator
        def test_function(array: np.ndarray) -> np.ndarray:
            if not isinstance(array, np.ndarray):
                raise ValueError("Input must be a numpy array")
            return array * 2

        result = test_function(self.torch_input.clone())
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cpu"
        if torch.cuda.is_available():
            result = test_function(self.torch_input.clone().to("cuda"))
            assert isinstance(result, torch.Tensor)
            assert result.device.type == "cuda"

    @pytest.mark.parametrize(  # type: ignore[misc]
        "input_data,expected_type",
        [
            ("numpy_input", np.ndarray),
            ("torch_input", torch.Tensor),
        ],
    )
    def test_array_conversion_decorator_with_tuples(
        self, input_data: str, expected_type: type
    ) -> None:
        """Test array_conversion_decorator with tuple return types."""

        @array_conversion_decorator
        def test_function(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            if not isinstance(array, np.ndarray):
                raise ValueError("Input must be a numpy array")
            return array * 2, array * 3

        result1, result2 = test_function(getattr(self, input_data))
        assert isinstance(result1, expected_type)
        assert isinstance(result2, expected_type)
        if isinstance(result1, np.ndarray):
            assert np.allclose(result1, np.array([2.0, 4.0, 6.0]))
            assert np.allclose(result2, np.array([3.0, 6.0, 9.0]))
        else:
            assert torch.equal(result1, torch.tensor([2.0, 4.0, 6.0]))
            assert torch.equal(result2, torch.tensor([3.0, 6.0, 9.0]))

    @pytest.mark.parametrize(  # type: ignore[misc]
        "input_data,expected_type",
        [
            ("numpy_input", np.ndarray),
            ("torch_input", torch.Tensor),
        ],
    )
    def test_array_conversion_decorator_with_lists(
        self, input_data: str, expected_type: type
    ) -> None:
        """Test array_conversion_decorator with list return types."""

        @array_conversion_decorator
        def test_function(array: np.ndarray) -> list[np.ndarray]:
            if not isinstance(array, np.ndarray):
                raise ValueError("Input must be a numpy array")
            return [array * 2, array * 3]

        results = test_function(getattr(self, input_data))
        assert isinstance(results, list)
        assert len(results) == 2
        assert isinstance(results[0], expected_type)
        assert isinstance(results[1], expected_type)
        if isinstance(results[0], np.ndarray):
            assert np.allclose(results[0], np.array([2.0, 4.0, 6.0]))
            assert np.allclose(results[1], np.array([3.0, 6.0, 9.0]))
        else:
            assert torch.equal(results[0], torch.tensor([2.0, 4.0, 6.0]))
            assert torch.equal(results[1], torch.tensor([3.0, 6.0, 9.0]))

    def test_array_conversion_decorator_no_tensor_args(self) -> None:
        """Test array_conversion_decorator with no tensor/array arguments."""

        @array_conversion_decorator
        def test_function(x: float, y: float) -> float:
            return x + y

        result = test_function(1.0, 2.0)
        assert isinstance(result, float)
        assert result == 3.0

    def test_array_conversion_decorator_none_result(self) -> None:
        """Test array_conversion_decorator with None result."""

        @array_conversion_decorator
        def test_function(array: np.ndarray) -> None:
            if not isinstance(array, np.ndarray):
                raise ValueError("Input must be a numpy array")
            return None

        result = test_function(self.numpy_input)
        assert result is None
