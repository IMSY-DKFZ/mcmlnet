"""Tests for mcmlnet.training.data_loading.data_augmentation_classes module."""

import pytest
import torch

from mcmlnet.training.data_loading.data_augmentation_classes import (
    BezierContrast,
    BrightDarkVariation,
    DataAugmentation,
    NoiseAddition,
    NoiseAdditionWithExponentialDecay,
    RandomOrder,
    ReflectanceMCNoise,
    ShotNoiseAddition,
)


class ConcreteDataAugmentation(DataAugmentation):
    """Concrete subclass of DataAugmentation for testing purposes."""

    def __init__(self, p: float) -> None:
        super().__init__(p)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data


class TestDataAugmentation:
    """Test cases for DataAugmentation base class."""

    def test_init_valid(self) -> None:
        """Test valid initialization."""
        aug = ConcreteDataAugmentation(p=0.5)

        assert aug.p == 0.5

    def test_init_invalid_p(self) -> None:
        """Test error with invalid probability."""
        with pytest.raises(
            ValueError, match="Probability 'p' must be a number between 0 and 1"
        ):
            ConcreteDataAugmentation(p=1.5)

        with pytest.raises(
            ValueError, match="Probability 'p' must be a number between 0 and 1"
        ):
            ConcreteDataAugmentation(p=-0.1)

    def test_call_not_implemented(self) -> None:
        """Test that __call__ raises NotImplementedError."""

        class ConcreteDataAugmentation(DataAugmentation):
            """Concrete subclass of DataAugmentation for testing purposes."""

            def __init__(self, p: float) -> None:
                super().__init__(p)

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ConcreteDataAugmentation(p=0.5)  # type: ignore[abstract]


class TestNoiseAddition:
    """Test cases for NoiseAddition class."""

    def test_init_valid(self) -> None:
        """Test valid initialization."""
        noise = NoiseAddition(snr=0.1, p=0.5)

        assert noise.p == 0.5
        assert noise.snr == 0.1

    def test_init_invalid_snr(self) -> None:
        """Test error with invalid snr."""
        with pytest.raises(ValueError, match="SNR must be positive"):
            NoiseAddition(snr=0, p=0.5)
        with pytest.raises(TypeError, match="SNR must be a numerical type"):
            NoiseAddition(snr="invalid", p=0.5)  # type: ignore[arg-type]

    def test_call_always_apply(self) -> None:
        """Test __call__ when probability is met."""
        noise = NoiseAddition(snr=0.1, p=1.0)
        data = torch.randn(10, 5)
        original_data = data.clone()

        result = noise(data)

        assert not torch.allclose(result, original_data)
        assert result.shape == original_data.shape  # type: ignore[union-attr]

    def test_call_never_apply(self) -> None:
        """Test __call__ when probability is not met."""
        noise = NoiseAddition(snr=0.1, p=0.0)
        data = torch.randn(10, 5)
        original_data = data.clone()

        result = noise(data)

        assert torch.allclose(result, original_data)
        assert result.shape == original_data.shape  # type: ignore[union-attr]

    def test_call_with_none_input(self) -> None:
        """Test __call__ with None input."""
        noise = NoiseAddition(snr=0.1, p=1.0)

        result = noise(None)
        assert result is None

    def test_call_invalid_data_type(self) -> None:
        """Test error with invalid data type."""
        noise = NoiseAddition(snr=0.1, p=0.5)

        with pytest.raises(TypeError, match=r"x must be torch.Tensor or np.ndarray"):
            noise([1, 2, 3])


class TestNoiseAdditionWithExponentialDecay:
    """Test cases for NoiseAdditionWithExponentialDecay class."""

    def test_init_valid(self) -> None:
        """Test valid initialization."""
        noise = NoiseAdditionWithExponentialDecay(
            snr=0.1, p=0.5, max_epochs=1000, batches_per_epoch=100
        )

        assert noise.p == 0.5
        assert noise.snr == 0.1
        assert noise.batches_per_epoch == 100

    def test_init_invalid_batches_per_epoch(self) -> None:
        """Test error with invalid batches_per_epoch."""
        with pytest.raises(ValueError, match="batches_per_epoch must be positive"):
            NoiseAdditionWithExponentialDecay(
                snr=0.1, p=0.5, max_epochs=1000, batches_per_epoch=0
            )

    def test_init_invalid_decay_scale(self) -> None:
        """Test error with invalid decay scale."""
        with pytest.raises(ValueError, match="decay_scale must be positive"):
            NoiseAdditionWithExponentialDecay(
                snr=0.1, p=0.5, max_epochs=1000, batches_per_epoch=100, decay_scale=0.0
            )

    def test_call_always_apply(self) -> None:
        """Test __call__ when probability is met."""
        noise = NoiseAdditionWithExponentialDecay(
            snr=0.1,
            p=1.0,
            max_epochs=1000,
            batches_per_epoch=100,
        )
        data = torch.randn(10, 5)
        original_data = data.clone()

        result = noise(data)

        assert not torch.allclose(result, original_data)
        assert result.shape == original_data.shape  # type: ignore[union-attr]

    def test_call_never_apply(self) -> None:
        """Test __call__ when probability is not met."""
        noise = NoiseAdditionWithExponentialDecay(
            snr=0.1, p=0.0, max_epochs=1000, batches_per_epoch=100
        )
        data = torch.randn(10, 5)
        original_data = data.clone()

        result = noise(data)

        assert torch.allclose(result, original_data)
        assert result.shape == original_data.shape  # type: ignore[union-attr]

    def test_call_with_none_input(self) -> None:
        """Test __call__ with None input."""
        noise = NoiseAdditionWithExponentialDecay(
            snr=0.1, p=1.0, max_epochs=1000, batches_per_epoch=100
        )

        result = noise(None)
        assert result is None

    def test_call_invalid_data_type(self) -> None:
        """Test error with invalid data type."""
        noise = NoiseAdditionWithExponentialDecay(
            snr=0.1, p=0.5, max_epochs=1000, batches_per_epoch=100
        )

        with pytest.raises(TypeError, match=r"x must be torch.Tensor or np.ndarray"):
            noise([1, 2, 3])


class TestReflectanceMCNoise:
    """Test cases for ReflectanceMCNoise class."""

    def test_init_valid(self) -> None:
        """Test valid initialization."""
        noise = ReflectanceMCNoise(n_photons=1000, p=0.5)

        assert noise.p == 0.5
        assert noise.n_photons == 1000

    def test_init_invalid_n_photons(self) -> None:
        """Test error with invalid n_photons."""
        with pytest.raises(ValueError, match="n_photons must be positive"):
            ReflectanceMCNoise(n_photons=0, p=0.5)

    def test_call_always_apply(self) -> None:
        """Test __call__ when probability is met."""
        noise = ReflectanceMCNoise(n_photons=1000, p=0.5)
        data = torch.randn(10, 5).clamp(1e-7, 1 - 1e-7)
        original_data = data.clone()

        result = noise(data)

        assert not torch.allclose(result, original_data)
        assert result.shape == original_data.shape  # type: ignore[union-attr]

    def test_call_never_apply(self) -> None:
        """Test __call__ when probability is not met."""
        noise = ReflectanceMCNoise(n_photons=1000, p=0.0)
        data = torch.randn(10, 5).clamp(1e-7, 1 - 1e-7)
        original_data = data.clone()

        result = noise(data)

        assert torch.allclose(result, original_data)
        assert result.shape == original_data.shape  # type: ignore[union-attr]

    def test_call_with_none_input(self) -> None:
        """Test __call__ with None input."""
        noise = ReflectanceMCNoise(n_photons=1000, p=0.5)

        result = noise(None)
        assert result is None


class TestShotNoiseAddition:
    """Test cases for ShotNoiseAddition class."""

    def test_init_valid(self) -> None:
        """Test valid initialization."""
        noise = ShotNoiseAddition(snr=20.0, p=0.5)

        assert noise.p == 0.5
        assert noise.snr == 20.0

    def test_init_invalid_snr_type(self) -> None:
        """Test error with invalid snr type."""
        with pytest.raises(TypeError, match="SNR must be a numerical type"):
            ShotNoiseAddition(snr="invalid", p=0.5)  # type: ignore[arg-type]

    def test_init_invalid_snr(self) -> None:
        """Test error with invalid snr."""
        with pytest.raises(ValueError, match="SNR must be positive"):
            ShotNoiseAddition(snr=0, p=0.5)

    def test_init_invalid_white_type(self) -> None:
        """Test error with invalid white type."""
        with pytest.raises(TypeError, match="white must be torch.Tensor"):
            ShotNoiseAddition(white=[1, 2, 3], dark=None, snr=20.0, p=0.5)

    def test_init_invalid_dark_type(self) -> None:
        """Test error with invalid dark type."""
        with pytest.raises(TypeError, match="dark must be torch.Tensor"):
            ShotNoiseAddition(white=None, dark=[1, 2, 3], snr=20.0, p=0.5)

    def test_call_always_apply(self) -> None:
        """Test __call__ when probability is met."""
        white_ref = torch.ones(5) * 0.9
        dark_ref = torch.ones(5) * 0.1
        noise = ShotNoiseAddition(white=white_ref, dark=dark_ref, snr=20.0, p=1.0)
        data = torch.randn(10, 5)
        original_data = data.clone()

        result = noise(data)

        assert not torch.allclose(result, original_data)
        assert result.shape == original_data.shape  # type: ignore[union-attr]

    def test_call_never_apply(self) -> None:
        """Test __call__ when probability is not met."""
        white_ref = torch.ones(5) * 0.9
        dark_ref = torch.ones(5) * 0.1
        noise = ShotNoiseAddition(white=white_ref, dark=dark_ref, snr=20.0, p=0.0)
        data = torch.randn(10, 5)
        original_data = data.clone()

        result = noise(data)

        assert torch.allclose(result, original_data)
        assert result.shape == original_data.shape  # type: ignore[union-attr]

    def test_call_with_none_input(self) -> None:
        """Test __call__ with None input."""
        white_ref = torch.ones(5) * 0.9
        dark_ref = torch.ones(5) * 0.1
        noise = ShotNoiseAddition(white=white_ref, dark=dark_ref, snr=20.0, p=1.0)

        result = noise(None)
        assert result is None


class TestBrightDarkVariation:
    """Test cases for BrightDarkVariation class."""

    def test_init_valid(self) -> None:
        """Test valid initialization."""
        variation = BrightDarkVariation(p=0.5, eps=0.2)

        assert variation.p == 0.5
        assert variation.eps == 0.2

    def test_init_invalid_eps_type(self) -> None:
        """Test error with invalid eps type."""
        with pytest.raises(TypeError, match="eps must be a numerical type"):
            BrightDarkVariation(p=0.5, eps="invalid")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="eps must be a numerical type"):
            BrightDarkVariation(p=0.5, eps=True)

    def test_init_invalid_eps(self) -> None:
        """Test error with invalid eps."""
        with pytest.raises(ValueError, match="eps must be positive"):
            BrightDarkVariation(p=0.5, eps=0)

    def test_call_always_apply(self) -> None:
        """Test __call__ when probability is met."""
        variation = BrightDarkVariation(p=1.0, eps=0.2)
        data = torch.randn(10, 5)
        original_data = data.clone()

        result = variation(data)

        assert not torch.allclose(result, original_data)
        assert result.shape == original_data.shape  # type: ignore[union-attr]

    def test_call_never_apply(self) -> None:
        """Test __call__ when probability is not met."""
        variation = BrightDarkVariation(p=0.0, eps=0.2)
        data = torch.randn(10, 5)
        original_data = data.clone()

        result = variation(data)

        assert torch.allclose(result, original_data)
        assert result.shape == original_data.shape  # type: ignore[union-attr]

    def test_call_with_none_input(self) -> None:
        """Test __call__ with None input."""
        variation = BrightDarkVariation(p=1.0, eps=0.2)

        result = variation(None)
        assert result is None


class TestBezierContrast:
    """Test cases for BezierContrast class."""

    def test_init_valid(self) -> None:
        """Test valid initialization."""
        contrast = BezierContrast(p=0.5, p_flip=0.2)

        assert contrast.p == 0.5
        assert contrast.p_flip == 0.2

    def test_init_invalid_p_flip_type(self) -> None:
        """Test error with invalid p_flip type."""
        with pytest.raises(TypeError, match="p_flip must be a numerical type"):
            BezierContrast(p=0.5, p_flip="invalid")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="p_flip must be a numerical type"):
            BezierContrast(p=0.5, p_flip=True)

    def test_init_invalid_p_flip(self) -> None:
        """Test error with invalid p_flip."""
        with pytest.raises(ValueError, match="p_flip must be between 0 and 1"):
            BezierContrast(p=0.5, p_flip=1.5)

    def test_call_always_apply(self) -> None:
        """Test __call__ when probability is met."""
        contrast = BezierContrast(p=1.0, p_flip=0.2)
        data = torch.randn(10, 5)
        original_data = data.clone()

        result = contrast(data)

        assert not torch.allclose(result, original_data)
        assert result.shape == original_data.shape  # type: ignore[union-attr]

    def test_call_never_apply(self) -> None:
        """Test __call__ when probability is not met."""
        contrast = BezierContrast(p=0.0, p_flip=0.2)
        data = torch.randn(10, 5)
        original_data = data.clone()

        result = contrast(data)

        assert torch.allclose(result, original_data)
        assert result.shape == original_data.shape  # type: ignore[union-attr]

    def test_call_with_none_input(self) -> None:
        """Test __call__ with None input."""
        contrast = BezierContrast(p=1.0, p_flip=0.2)

        result = contrast(None)
        assert result is None


class TestDataAugmentationIntegration:
    """Integration tests for data augmentation classes."""

    def test_multiple_augmentations_chain(self) -> None:
        """Test chaining multiple augmentations."""
        noise = NoiseAddition(snr=0.1, p=1.0)
        variation = BrightDarkVariation(p=1.0, eps=0.2)

        data = torch.randn(10, 5)
        original_data = data.clone()

        result = variation(noise(data))

        assert not torch.allclose(result, original_data)
        assert result.shape == original_data.shape  # type: ignore[union-attr]

    def test_random_order_with_mixed_probabilities(self) -> None:
        """Test RandomOrder with mixed probabilities."""
        transforms = [
            NoiseAddition(snr=0.1, p=0.5),
            BrightDarkVariation(p=0.3, eps=0.2),
        ]
        random_order = RandomOrder(transforms)

        data = torch.randn(10, 5)

        # Run multiple times to test randomness
        results = []
        for _ in range(5):
            result = random_order(data)
            results.append(result.clone())

        # At least some results should be different due to randomness
        unique_results = len(
            {torch.allclose(r1, r2) for r1 in results for r2 in results}
        )
        assert unique_results > 1

    def test_augmentation_with_edge_cases(self) -> None:
        """Test augmentations with edge cases."""
        # Test with single sample
        noise = NoiseAddition(snr=0.1, p=1.0)
        data = torch.randn(1, 5)
        result = noise(data)
        assert result.shape == (1, 5)  # type: ignore[union-attr]

        # Test with single wavelength
        data = torch.randn(10, 1)
        result = noise(data)
        assert result.shape == (10, 1)  # type: ignore[union-attr]

        # Even with large SNR (small denominator) function should complete successfully
        noise = NoiseAddition(snr=1.0, p=1.0, clipping=False)
        data = torch.randn(10, 5)
        result = noise(data)
        assert result.shape == (10, 5)  # type: ignore[union-attr]

        noise = NoiseAddition(snr=1.0, p=1.0, clipping=True)
        data = torch.randn(10, 5)
        result = noise(data)
        assert result.shape == (10, 5)  # type: ignore[union-attr]
        assert torch.all(result >= 0) and torch.all(result <= 1)  # type: ignore[operator]

    def test_augmentation_preserves_dtype(self) -> None:
        """Test that augmentations preserve data type."""
        noise = NoiseAddition(snr=0.1, p=1.0)
        data = torch.randn(10, 5, dtype=torch.float32)

        result = noise(data)
        assert result.dtype == torch.float32  # type: ignore[union-attr]

        # Test with double precision
        data = torch.randn(10, 5, dtype=torch.float64)
        result = noise(data)
        assert result.dtype == torch.float64  # type: ignore[union-attr]
