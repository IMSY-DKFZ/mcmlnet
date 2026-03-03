"""Tests for mcmlnet.utils.mc_runner module."""

import copy
import os
import shutil
import tempfile
import warnings
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from mcmlnet.utils.mc_runner import RecomputeMC


class TestRecomputeMC:
    """Test cases for RecomputeMC class."""

    def setup_method(self) -> None:
        """Setup method to initialize common variables."""
        self.wavelengths = np.array([400e-9, 500e-9, 600e-9])
        self.df = pd.DataFrame(np.random.randn(100, 5))
        self.batch_size = 25
        self.data_array = np.random.uniform(size=(10, 24))  # 8 params * 3 layers
        self.data_array_valid = copy.deepcopy(self.data_array)
        self.data_array_valid[:, [2, 10, 18]] *= 5000
        self.data_array_valid[:, [4, 12, 20]] = 0
        self.data_array_valid[:, [6, 14, 22]] += 1
        self.data_array_valid[:, [7, 15, 23]] /= 100
        self.data_array_valid[:, -1] = 0.2

    def test_init_valid_wavelengths(self) -> None:
        """Test initialization with valid wavelengths."""
        mc_runner = RecomputeMC(wavelengths=self.wavelengths)

        assert np.array_equal(mc_runner.wavelengths, self.wavelengths)
        assert mc_runner.nr_photons == 10**6
        assert mc_runner.ignore_a is True
        assert mc_runner.mco_folder == ""
        assert mc_runner.verbose is False
        assert mc_runner.timeout is None

    def test_init_invalid_wavelengths(self) -> None:
        """Test initialization with invalid wavelengths (out of range)."""
        wavelengths = self.wavelengths.copy()
        wavelengths[0] = 100e-9
        with pytest.raises(ValueError, match="Wavelengths must be in the range"):
            RecomputeMC(wavelengths=wavelengths)

        wavelengths = self.wavelengths.copy()
        wavelengths[0] = 2500e-9
        with pytest.raises(ValueError, match="Wavelengths must be in the range"):
            RecomputeMC(wavelengths=wavelengths)

    def test_init_custom_parameters(self) -> None:
        """Test initialization with custom parameters."""
        mc_runner = RecomputeMC(
            wavelengths=self.wavelengths,
            nr_photons=10**5,
            ignore_a=False,
            mco_folder="/path/to/mco",
            verbose=True,
            timeout=30.0,
        )

        assert mc_runner.nr_photons == 10**5
        assert mc_runner.ignore_a is False
        assert mc_runner.mco_folder == "/path/to/mco"
        assert mc_runner.verbose is True
        assert mc_runner.timeout == 30.0

    def test_warn_if_batch_not_full_valid_batch_size(self) -> None:
        """Test no warning when batch size divides sample count exactly."""
        with warnings.catch_warnings():
            RecomputeMC._warn_if_batch_not_full(self.df, self.batch_size)
            warnings.simplefilter("error")

    def test_warn_if_batch_not_full(self) -> None:
        """Test warning when batch size doesn't divide sample count."""
        batch_size = 30

        # Should warn about 10 ignored samples (100 % 30 = 10)
        with pytest.warns(
            UserWarning,
            match=(
                "Batch size does not divide the number of samples, "
                "the last 10 samples will be ignored!"
            ),
        ):
            RecomputeMC._warn_if_batch_not_full(self.df, batch_size)

    def test_determine_batch_range_default(self) -> None:
        """Test batch range determination with default end value."""
        batch_range = (0, -1)

        start, end = RecomputeMC._determine_batch_range(
            self.df, self.batch_size, batch_range
        )
        assert start == 0
        assert end == 4

    def test_determine_batch_range_custom(self) -> None:
        """Test batch range determination with custom range."""
        batch_range = (1, 3)

        start, end = RecomputeMC._determine_batch_range(
            self.df, self.batch_size, batch_range
        )
        assert start == 1
        assert end == 3

    def test_generate_column_headers(self) -> None:
        """Test column header generation."""
        columns = ["sao2", "vhb", "a_mie"]
        n_layers = 2

        top_level, bottom_level = RecomputeMC.generate_column_headers(columns, n_layers)

        expected_top = np.array(
            ["layer0", "layer0", "layer0", "layer1", "layer1", "layer1"]
        )
        expected_bottom = np.array(["sao2", "vhb", "a_mie", "sao2", "vhb", "a_mie"])

        assert np.array_equal(top_level, expected_top)
        assert np.array_equal(bottom_level, expected_bottom)

    def test_array_to_mc_sim_df_valid_shape(self) -> None:
        """Test conversion of valid array to DataFrame."""
        mc_runner = RecomputeMC(wavelengths=self.wavelengths)

        # Create array with shape (n_samples, 24) - 8 params * 3 layers
        df = mc_runner.array_to_mc_sim_df(self.data_array.copy())

        assert df.shape == self.data_array.shape
        assert isinstance(df, pd.DataFrame)
        assert df.columns.nlevels == 2

    def test_array_to_mc_sim_df_invalid_shape(self) -> None:
        """Test conversion of invalid array shape."""
        mc_runner = RecomputeMC(wavelengths=self.wavelengths)

        # Create array with wrong shape
        array = np.random.randn(10, 20)

        with pytest.raises(
            AssertionError, match="Optical parameter array has to be of shape"
        ):
            mc_runner.array_to_mc_sim_df(array)

    def test_csv_to_mc_sim_df_from_path(self) -> None:
        """Test loading DataFrame from CSV path."""
        mc_runner = RecomputeMC(wavelengths=self.wavelengths)

        # Create temporary CSV file with correct header and data
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            # Write header
            f.write(str([f"col_{i}" for i in range(24)]) + "\n")
            # Write five rows of data
            for i, _ in enumerate(self.data_array.copy()):
                f.write(
                    f"{i},"
                    + ",".join([str(np.random.rand()) for _ in range(24)])
                    + "\n"
                )
            temp_path = f.name

        try:
            df = mc_runner.csv_to_mc_sim_df(path=temp_path)
            assert df.shape == self.data_array.shape
            assert isinstance(df, pd.DataFrame)
        finally:
            os.unlink(temp_path)

    def test_csv_to_mc_sim_df_from_dataframe(self) -> None:
        """Test loading DataFrame from existing DataFrame."""
        mc_runner = RecomputeMC(wavelengths=self.wavelengths)

        # Create DataFrame with correct structure
        input_df = pd.DataFrame(
            self.data_array.copy(), columns=[f"col_{i}" for i in range(24)]
        )

        df = mc_runner.csv_to_mc_sim_df(df=input_df)
        assert df.shape == self.data_array.shape
        assert isinstance(df, pd.DataFrame)

    def test_csv_to_mc_sim_df_missing_input(self) -> None:
        """Test error when neither df nor path is provided."""
        mc_runner = RecomputeMC(wavelengths=self.wavelengths)

        with pytest.raises(
            ValueError,
            match=("Missing Input: Either 'df' or 'path' have to be defined."),
        ):
            mc_runner.csv_to_mc_sim_df()

    def test_convert_batch_to_df_dataframe(self) -> None:
        """Test conversion of DataFrame input."""
        mc_runner = RecomputeMC(wavelengths=self.wavelengths)

        input_df = pd.DataFrame(self.data_array.copy())
        result = mc_runner.convert_batch_to_df(input_df)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == input_df.shape
        assert np.all(result.values == self.data_array)

    def test_convert_batch_to_df_array(self) -> None:
        """Test conversion of numpy array input."""
        mc_runner = RecomputeMC(wavelengths=self.wavelengths)

        result = mc_runner.convert_batch_to_df(self.data_array.copy())

        assert isinstance(result, pd.DataFrame)
        assert result.shape == self.data_array.shape
        assert np.all(result.values == self.data_array)

    def test_convert_batch_to_df_string(self) -> None:
        """Test conversion of string path input."""
        mc_runner = RecomputeMC(wavelengths=self.wavelengths)

        # Create temporary CSV file with additional unnamed index column
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Unnamed: 0," + ",".join([f"col_{i}" for i in range(24)]) + "\n")
            for i, _ in enumerate(self.data_array.copy()):
                f.write(
                    f"{i},"
                    + ",".join([str(np.random.rand()) for _ in range(24)])
                    + "\n"
                )
            temp_path = f.name

        try:
            result = mc_runner.convert_batch_to_df(temp_path)
            assert isinstance(result, pd.DataFrame)
            assert result.shape == self.data_array.shape
        finally:
            os.unlink(temp_path)

    @pytest.mark.parametrize(  # type: ignore[misc]
        "invalid_input", [123, 45.6, [1, 2, 3], None, torch.randn([20, 24])]
    )
    def test_convert_batch_to_df_invalid_type(self, invalid_input: Any) -> None:
        """Test error with invalid input type."""
        mc_runner = RecomputeMC(wavelengths=self.wavelengths)

        with pytest.raises(TypeError, match="Unknown type"):
            mc_runner.convert_batch_to_df(invalid_input)

    @pytest.mark.parametrize(  # type: ignore[misc]
        "batch_size,batch_id",
        list(
            product(
                [10, 20, 25, 100],
                [0, 1, 2, 3],
            )
        ),
    )
    def test_load_or_simulate_batch_file_exists(
        self, batch_size: int, batch_id: int
    ) -> None:
        """Test loading existing batch file."""
        mc_runner = RecomputeMC(wavelengths=self.wavelengths)

        with tempfile.TemporaryDirectory() as temp_dir:
            batch_file = os.path.join(temp_dir, f"batch_{batch_id}.csv")
            columns = mc_runner.generate_column_headers(
                ["sao2", "vhb", "a_mie", "b_mie", "a_ray", "g", "n", "d"], 3
            )
            batch_data = pd.DataFrame(self.data_array.copy(), columns=columns)
            batch_data.to_csv(batch_file, index=False)

            input_batch = pd.DataFrame(np.random.randn(batch_size, 24))

            result = mc_runner._load_or_simulate_batch(
                input_batch, temp_dir, len(self.data_array), batch_id
            )

            assert isinstance(result, pd.DataFrame)
            assert result.shape == self.data_array.shape
            assert np.allclose(result.values, self.data_array)

    def test_load_or_simulate_batch_file_exists_wrong_batch_size(self) -> None:
        """Test loading existing batch file with wrong batch size."""
        mc_runner = RecomputeMC(wavelengths=self.wavelengths)

        with tempfile.TemporaryDirectory() as temp_dir:
            batch_file = os.path.join(temp_dir, "batch_0.csv")
            columns = mc_runner.generate_column_headers(
                ["sao2", "vhb", "a_mie", "b_mie", "a_ray", "g", "n", "d"], 3
            )
            batch_data = pd.DataFrame(self.data_array.copy(), columns=columns)
            batch_data.to_csv(batch_file, index=False)

            input_batch = pd.DataFrame(np.random.randn(20, 24))

            with pytest.raises(
                AssertionError,
                match=(
                    "Batch size does not match. "
                    "Check given batch size and number of samples."
                ),
            ):
                mc_runner._load_or_simulate_batch(input_batch, temp_dir, 20, 0)

    @pytest.mark.skipif(  # type: ignore[misc]
        not torch.cuda.is_available() or shutil.which("MCML") is None,
        reason="Requires GPU and MCML installation",
    )
    def test_load_or_simulate_batch_file_not_exists(self) -> None:
        """Test simulating batch when file doesn't exist."""
        mc_runner = RecomputeMC(wavelengths=self.wavelengths)

        with tempfile.TemporaryDirectory() as temp_dir:
            columns = mc_runner.generate_column_headers(
                ["sao2", "vhb", "a_mie", "b_mie", "a_ray", "g", "n", "d"], 3
            )
            input_batch = pd.DataFrame(self.data_array.copy(), columns=columns)
            result = mc_runner._load_or_simulate_batch(input_batch, temp_dir, 10, 0)

            assert isinstance(result, pd.DataFrame)
            assert result.shape == (
                self.data_array.shape[0],
                self.data_array.shape[1] + len(self.wavelengths) * 7,
            )

    @pytest.mark.skipif(  # type: ignore[misc]
        not torch.cuda.is_available() or shutil.which("MCML") is None,
        reason="Requires GPU and MCML installation",
    )
    def test_run_simulation_from_df(self) -> None:
        """Test running simulation from DataFrame."""
        mc_runner = RecomputeMC(wavelengths=self.wavelengths)

        with tempfile.TemporaryDirectory() as temp_dir:
            columns = mc_runner.generate_column_headers(
                ["sao2", "vhb", "a_mie", "b_mie", "a_ray", "g", "n", "d"], 3
            )
            input_batch = pd.DataFrame(self.data_array.copy(), columns=columns)
            result = mc_runner.run_simulation_from_df(
                input_batch, temp_dir, batch_size=2, batch_range=(0, 2)
            )

            assert isinstance(result, np.ndarray)
            assert result.shape == (2 + 2, len(self.wavelengths))
            assert np.all(result >= 0) and np.all(result <= 1)

    @pytest.mark.skipif(  # type: ignore[misc]
        not torch.cuda.is_available() or shutil.which("MCML") is None,
        reason="Requires GPU and MCML installation",
    )
    def test_run_simulation_from_df_creates_directory(self) -> None:
        """Test that simulation creates directory if it doesn't exist."""
        mc_runner = RecomputeMC(wavelengths=self.wavelengths)

        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = os.path.join(temp_dir, "new_sim_dir/")
            _ = pd.DataFrame(np.random.randn(20, 24))

            columns = mc_runner.generate_column_headers(
                ["sao2", "vhb", "a_mie", "b_mie", "a_ray", "g", "n", "d"], 3
            )
            input_batch = pd.DataFrame(self.data_array.copy(), columns=columns)
            _ = mc_runner.run_simulation_from_df(
                input_batch, new_dir, batch_size=10, batch_range=(0, 2)
            )
            assert os.path.exists(new_dir)
