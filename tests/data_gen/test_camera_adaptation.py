import os

import numpy as np
import pandas as pd
import pytest

from mcmlnet.data_gen.camera_adaptation import adapt_simulations_to_tivita


@pytest.fixture  # type: ignore[misc]
def dummy_filters() -> pd.DataFrame:
    """Create a dummy filter response CSV file."""
    return pd.DataFrame(
        np.concatenate([np.eye(5), np.ones((95, 5))], axis=0),
        columns=["500e-9", "600e-9", "700e-9", "800e-9", "900e-9"],
    )


@pytest.fixture  # type: ignore[misc]
def dummy_irradiance() -> pd.DataFrame:
    """Create a dummy irradiance data file."""
    return pd.DataFrame(
        {
            ">>>>>Begin": [500, 600, 700, 800, 900],
            "Spectral": [10.0, 20.0, 30.0, 40.0, 50.0],
            "Data<<<<<": [0, 0, 0, 0, 0],
        }
    )


@pytest.fixture  # type: ignore[misc]
def dummy_simulations() -> np.ndarray:
    """Create dummy simulation data."""
    return np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]])


@pytest.fixture  # type: ignore[misc]
def dummy_wavelengths() -> np.ndarray:
    """Create dummy wavelength data."""
    return np.array([500.0, 600.0, 700.0, 800.0, 900.0]) * 1e-9


def test_adapt_simulations_to_tivita(
    dummy_filters: pd.DataFrame,
    dummy_irradiance: pd.DataFrame,
    dummy_simulations: np.ndarray,
    dummy_wavelengths: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the adaptation of simulations to TIVITA camera wavelengths."""

    def fake_read_csv(path: str, *args, **kwargs) -> pd.DataFrame:  # type: ignore[no-untyped-def]
        if "artificial_tivita_camera_uniform.csv" in path:
            return dummy_filters
        elif "tivita_relative_irradiance_2019_04_05.txt" in path:
            return dummy_irradiance
        else:
            raise FileNotFoundError(path)

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    out = adapt_simulations_to_tivita(
        dummy_simulations,
        dummy_wavelengths,
        filter_type="uniform",
        cache_file_name="",
        cache_dir=os.environ["cache_dir"],
    )
    assert isinstance(out, np.ndarray)
    assert out.shape == (len(dummy_simulations), 100)
    filtered_reflectance = np.array([[0.35] * 95, [0.85] * 95])
    np.testing.assert_allclose(
        out, np.concatenate([dummy_simulations, filtered_reflectance], axis=1)
    )
