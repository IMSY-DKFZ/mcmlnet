"""Tests for mcmlnet.utils.loading module."""

import copy
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

from mcmlnet.utils.loading import SimulationDataLoader


class TestSimulationDataLoader:
    """Test cases for SimulationDataLoader class."""

    def setup_method(self) -> None:
        """Setup method to initialize common variables."""
        base_layer = pd.DataFrame(
            {
                ("layer0", "sao2"): [0.1, 0.2, 0.3],
                ("layer0", "vhb"): [0.1, 0.2, 0.3],
                ("layer0", "a_mie"): [1000, 2000, 3000],
                ("layer0", "b_mie"): [1.0, 2.0, 3.0],
                ("layer0", "a_ray"): [0, 0, 0.1],
                ("layer0", "g"): [0.5, 0.6, 0.7],
                ("layer0", "n"): [1.3, 1.4, 1.5],
                ("layer0", "d"): [0.1, 0.01, 0.2],
                ("layer0", "chb"): [100, 200, 300],  # To be dropped
            }
        )
        # Define valid single and three-layer DataFrames
        self.single_layer_df = base_layer.copy()
        self.single_layer_df[("reflectances", "300")] = [0.1, 0.2, 0.3]
        self.single_layer_df[("reflectances", "310")] = [0.15, 0.25, 0.35]
        layers = []
        for i in range(3):
            layer = base_layer.copy()
            layer.columns = pd.MultiIndex.from_tuples(
                [(f"layer{i}", col[1]) for col in layer.columns]
            )
            layers.append(layer)
        self.three_layer_df = pd.concat(layers, axis=1)
        self.three_layer_df[("reflectances", "300")] = [0.1, 0.2, 0.3]
        self.three_layer_df[("reflectances", "310")] = [0.15, 0.25, 0.35]
        # Define physical parameters (n_samples, n_wavelengths, n_params)
        self.n_samples = 10
        self.n_wavelengths = 3
        self.n_params_physical_short_format = 16  # 15 params + 1 reflectance
        self.physical_data = np.random.randn(
            self.n_samples, self.n_wavelengths, self.n_params_physical_short_format
        )

    def test_init_default_parameters(self) -> None:
        """Test initialization with default parameters."""
        loader = SimulationDataLoader()

        assert loader.n_wavelengths is None
        assert loader.n_layers is None
        assert loader.n_physio_params == 8
        assert loader.n_physical_params == 5
        assert loader.is_physical is False
        assert loader.simulation_df is None
        assert loader.simulations is None
        assert loader.simulation_tensor is None

    def test_init_custom_parameters(self) -> None:
        """Test initialization with custom parameters."""
        loader = SimulationDataLoader(
            n_wavelengths=100,
            n_layers=3,
            n_physio_params=6,
            n_physical_params=4,
            is_physical=True,
        )

        assert loader.n_wavelengths == 100
        assert loader.n_layers == 3
        assert loader.n_physio_params == 6
        assert loader.n_physical_params == 4
        assert loader.is_physical is True

    def test_get_columns_by_name(self) -> None:
        """Test getting columns by name."""
        df = pd.DataFrame(
            {
                "layer0_sao2": [1, 2, 3],
                "layer0_vhb": [4, 5, 6],
                "layer1_sao2": [7, 8, 9],
                "reflectances": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            }
        )

        layer0_cols = SimulationDataLoader._get_columns_by_name(df, "layer0")
        assert len(layer0_cols.columns) == 2
        assert "layer0_sao2" in layer0_cols.columns
        assert "layer0_vhb" in layer0_cols.columns

        df = pd.DataFrame(
            {
                ("layer0", "sao2"): [1, 2, 3],
                ("layer0", "vhb"): [4, 5, 6],
                ("layer1", "sao2"): [7, 8, 9],
                ("reflectances",): [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            }
        )

        layer0_cols = SimulationDataLoader._get_columns_by_name(df, "layer0")
        assert len(layer0_cols.columns) == 2
        assert isinstance(layer0_cols.columns, pd.MultiIndex)
        assert "sao2" in layer0_cols.layer0.columns
        assert "vhb" in layer0_cols.layer0.columns

    def test_drop_chb_columns_with_chb(self) -> None:
        """Test dropping chb columns when they exist."""
        df = pd.DataFrame(
            {"layer0_sao2": [1, 2, 3], "layer0_chb": [4, 5, 6], "layer0_vhb": [7, 8, 9]}
        )

        result = SimulationDataLoader._drop_columns(df)
        assert "layer0_chb" not in result.columns
        assert "layer0_sao2" in result.columns
        assert "layer0_vhb" in result.columns

    def test_drop_chb_columns_without_chb(self) -> None:
        """Test dropping chb columns when they don't exist."""
        df = pd.DataFrame({"layer0_sao2": [1, 2, 3], "layer0_vhb": [4, 5, 6]})

        result = SimulationDataLoader._drop_columns(df.copy())
        assert result.equals(df)

    def test_load_from_directory_success(self) -> None:
        """Test loading from directory with parquet files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test parquet files
            df1 = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
            df2 = pd.DataFrame({"col1": [7, 8, 9], "col2": [10, 11, 12]})

            df1.to_parquet(os.path.join(temp_dir, "file1.parquet"))
            df2.to_parquet(os.path.join(temp_dir, "file2.parquet"))

            result = SimulationDataLoader._load_from_directory(temp_dir)
            assert len(result) == 6  # 3 + 3 rows
            assert "col1" in result.columns
            assert "col2" in result.columns

    def test_load_from_directory_no_files(self) -> None:
        """Test loading from directory with no parquet files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(FileNotFoundError, match=r"No `.parquet` files found"):
                SimulationDataLoader._load_from_directory(temp_dir)

    def test_load_from_directory_mixed_files(self) -> None:
        """Test loading from directory with mixed file types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            df = pd.DataFrame({"col1": [1, 2, 3]})
            df.to_parquet(os.path.join(temp_dir, "file1.parquet"))

            with open(os.path.join(temp_dir, "file2.txt"), "w") as f:
                f.write("test")

            result = SimulationDataLoader._load_from_directory(temp_dir)
            assert len(result) == 3

    def test_simulation_to_standard_numpy_valid(self) -> None:
        """Test conversion to standard numpy with valid data."""
        loader = SimulationDataLoader()

        result = loader.simulation_to_standard_numpy(self.three_layer_df.copy())

        assert result.shape == (3, 26)  # 24 params + 2 wavelengths
        assert loader.n_wavelengths == 2
        assert loader.n_layers == 3

        loader = SimulationDataLoader()

        result = loader.simulation_to_standard_numpy(self.single_layer_df.copy())

        assert result.shape == (3, 10)  # 8 params + 2 wavelengths
        assert loader.n_wavelengths == 2
        assert loader.n_layers == 1

    def test_simulation_to_standard_numpy_wavelength_mismatch(self) -> None:
        """Test error when wavelength count doesn't match expected."""
        with pytest.raises(AssertionError, match="Wavelength mismatch"):
            loader = SimulationDataLoader(n_wavelengths=3)
            loader.simulation_to_standard_numpy(self.three_layer_df.copy())

        with pytest.raises(AssertionError, match="Wavelength mismatch"):
            loader = SimulationDataLoader(n_wavelengths=3)
            loader.simulation_to_standard_numpy(self.single_layer_df.copy())

    def test_simulation_to_standard_numpy_no_layers(self) -> None:
        """Test error when no layer data is found."""
        with pytest.raises(ValueError, match="No layer data found"):
            loader = SimulationDataLoader()
            loader.simulation_to_standard_numpy(self.three_layer_df.copy().layer0)

        with pytest.raises(ValueError, match="No layer data found"):
            loader = SimulationDataLoader()
            loader.simulation_to_standard_numpy(self.single_layer_df.copy().layer0)

    def test_simulation_to_standard_numpy_shape_mismatch(self) -> None:
        """Test error when data shape doesn't match expected."""
        loader = SimulationDataLoader()

        df = self.three_layer_df.copy()
        df[("layer0", "extra_col")] = [1, 2, 3]

        with pytest.raises(AssertionError, match="Shape mismatch"):
            loader.simulation_to_standard_numpy(df)

        loader = SimulationDataLoader()

        df = self.single_layer_df.copy()
        df[("layer0", "extra_col")] = [1, 2, 3]

        with pytest.raises(AssertionError, match="Shape mismatch"):
            loader.simulation_to_standard_numpy(df)

    def test_seeded_shuffle_idcs_tensor(self) -> None:
        """Test seeded shuffle with tensor input."""
        result = SimulationDataLoader.seeded_shuffle_idcs(torch.randn(10, 5), seed=42)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (10,)
        assert torch.all(result >= 0) and torch.all(result < 10)

        result = SimulationDataLoader.seeded_shuffle_idcs(
            torch.randn(10, 3, 5), seed=42
        )

        assert isinstance(result, torch.Tensor)
        assert result.shape == (10,)
        assert torch.all(result >= 0) and torch.all(result < 10)

    def test_seeded_shuffle_idcs_int(self) -> None:
        """Test seeded shuffle with integer input."""
        result = SimulationDataLoader.seeded_shuffle_idcs(10, seed=42)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (10,)
        assert torch.all(result >= 0) and torch.all(result < 10)

    def test_seeded_shuffle_idcs_reproducible(self) -> None:
        """Test that seeded shuffle is reproducible."""
        tensor = torch.randn(10, 5)

        result1 = SimulationDataLoader.seeded_shuffle_idcs(
            copy.deepcopy(tensor), seed=42
        )
        result2 = SimulationDataLoader.seeded_shuffle_idcs(
            copy.deepcopy(tensor), seed=42
        )

        assert torch.all(result1 == result2)

    def test_load_simulation_data_parquet_file(self) -> None:
        """Test loading simulation data from parquet file."""
        with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp_file:
            file_name = os.path.basename(tmp_file.name)
            folder = os.path.dirname(tmp_file.name)
            self.three_layer_df.copy().to_parquet(os.path.join(folder, file_name))

            loader = SimulationDataLoader()
            result = loader.load_simulation_data(file_name, folder)

            assert isinstance(result, torch.Tensor)
            assert result.shape == (
                self.three_layer_df.shape[0],
                26,
            )  # 24 params + 2 wavelengths

            self.single_layer_df.copy().to_parquet(os.path.join(folder, file_name))

            loader = SimulationDataLoader()
            result = loader.load_simulation_data(file_name, folder)

            assert isinstance(result, torch.Tensor)
            assert result.shape == (
                self.single_layer_df.shape[0],
                10,
            )  # 8 params + 2 wavelengths

    def test_load_simulation_data_csv_file(self) -> None:
        """Test loading simulation data from CSV file."""
        with tempfile.NamedTemporaryFile(suffix=".csv") as tmp_file:
            file_name = os.path.basename(tmp_file.name)
            folder = os.path.dirname(tmp_file.name)
            df = self.three_layer_df.copy()
            df.to_csv(os.path.join(folder, file_name), index=False)

            loader = SimulationDataLoader()
            result = loader.load_simulation_data(file_name, folder)

            assert isinstance(result, torch.Tensor)
            assert result.shape == (
                self.three_layer_df.shape[0],
                26,
            )  # 24 params + 2 wavelengths

            df = self.single_layer_df.copy()
            df.to_csv(os.path.join(folder, file_name), index=False)

            loader = SimulationDataLoader()
            result = loader.load_simulation_data(file_name, folder)

            assert isinstance(result, torch.Tensor)
            assert result.shape == (
                self.single_layer_df.shape[0],
                10,
            )  # 8 params + 2 wavelengths

    def test_load_simulation_data_directory(self) -> None:
        """Test loading simulation data from directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            df1, df2 = self.three_layer_df.copy(), self.three_layer_df.copy()
            os.makedirs(os.path.join(temp_dir, "test_data"), exist_ok=True)

            df1.to_parquet(os.path.join(temp_dir, "test_data/file1.parquet"))
            df2.to_parquet(os.path.join(temp_dir, "test_data/file2.parquet"))

            loader = SimulationDataLoader()
            result = loader.load_simulation_data("test_data", temp_dir)

            assert isinstance(result, torch.Tensor)
            assert result.shape == (6, 26)  # 3 + 3 rows, 26 cols

            df1, df2 = self.single_layer_df.copy(), self.single_layer_df.copy()

            df1.to_parquet(os.path.join(temp_dir, "test_data/file1.parquet"))
            df2.to_parquet(os.path.join(temp_dir, "test_data/file2.parquet"))

            loader = SimulationDataLoader()
            result = loader.load_simulation_data("test_data", temp_dir)

            assert isinstance(result, torch.Tensor)
            assert result.shape == (6, 10)  # 3 + 3 rows, 10 cols

    def test_load_simulation_data_file_not_found(self) -> None:
        """Test error when simulation data file is not found."""
        loader = SimulationDataLoader()

        with pytest.raises(FileNotFoundError, match="Dataset test_file not found"):
            loader.load_simulation_data("test_file")

    def test_load_physical_simulation_data_short_format(self) -> None:
        """Test loading physical simulation data in short format."""
        loader = SimulationDataLoader()

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            file_name = os.path.basename(f.name)
            folder = os.path.dirname(f.name)
            # Reshape to short 2D for parquet
            data_2d = copy.deepcopy(self.physical_data).reshape(
                -1, self.n_params_physical_short_format
            )
            pd.DataFrame(data_2d).to_parquet(os.path.join(folder, file_name))

            result = loader.load_physical_simulation_data(
                file_name, self.n_wavelengths, folder
            )

            assert isinstance(result, torch.Tensor)
            assert loader.is_physical is True
            assert result.shape == (
                self.n_samples,
                self.n_wavelengths,
                self.n_params_physical_short_format,
            )

    def test_load_physical_simulation_data_long_format(self) -> None:
        """Test loading physical simulation data in long format."""
        loader = SimulationDataLoader()

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            file_name = os.path.basename(f.name)
            folder = os.path.dirname(f.name)
            # Reshape to long 2D for parquet
            data_2d = copy.deepcopy(self.physical_data).reshape(
                len(self.physical_data), -1
            )
            pd.DataFrame(data_2d).to_parquet(os.path.join(folder, file_name))

            result = loader.load_physical_simulation_data(
                file_name, self.n_wavelengths, folder
            )

            assert isinstance(result, torch.Tensor)
            assert loader.is_physical is True
            assert loader.n_layers == 3

    def test_load_physical_simulation_data_wavelength_mismatch(self) -> None:
        """Test error when wavelength count doesn't match."""
        loader = SimulationDataLoader(n_wavelengths=3)

        data = np.random.randn(5, 10)  # Wrong shape

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            file_name = os.path.basename(f.name)
            folder = os.path.dirname(f.name)
            pd.DataFrame(data).to_parquet(os.path.join(folder, file_name))

            with pytest.raises(AssertionError, match="Wavelength mismatch"):
                loader.load_physical_simulation_data(file_name, 2, folder)

    def test_load_physical_simulation_data_shape_mismatch(self) -> None:
        """Test error when data shape is invalid."""
        loader = SimulationDataLoader()

        data = np.random.randn(5, 7)  # Invalid shape

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            file_name = os.path.basename(f.name)
            folder = os.path.dirname(f.name)
            pd.DataFrame(data).to_parquet(os.path.join(folder, file_name))

            with pytest.raises(AssertionError, match="Shape mismatch"):
                loader.load_physical_simulation_data(file_name, 2, folder)

    def test_load_physical_simulation_data_layer_mismatch(self) -> None:
        """Test error when layer count is invalid."""
        loader = SimulationDataLoader(n_layers=2)

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            file_name = os.path.basename(f.name)
            folder = os.path.dirname(f.name)
            data_2d = copy.deepcopy(self.physical_data).reshape(
                len(self.physical_data), -1
            )
            pd.DataFrame(data_2d).to_parquet(os.path.join(folder, file_name))

            with pytest.raises(AssertionError, match="Layer mismatch"):
                loader.load_physical_simulation_data(
                    file_name, self.n_wavelengths, folder
                )
