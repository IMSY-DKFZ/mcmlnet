"""Tests for mcmlnet.transforms.related_work_transformations module."""

import os
import tempfile

import numpy as np
import pytest
import torch

from mcmlnet.transforms.related_work_transformations import (
    LanConstants,
    LanTransformation,
    ManojlovicConstants,
    ManojlovicTransformation,
    Transformation,
    TsuiConstants,
    TsuiTransformation,
    get_bilirubin_extended_upper_range,
    get_mu_a_collagen,
    get_mu_a_epidermis,
    get_mu_a_water,
    get_oxidised_cytochrome_c,
    get_reduced_cytochrome_c,
    validate_file_and_load,
)


class TestValidateFileAndLoad:
    """Test cases for validate_file_and_load function."""

    def test_file_not_found(self) -> None:
        """Test file not found error."""
        with pytest.raises(FileNotFoundError, match="Reference file not found"):
            validate_file_and_load("/not/a/real/file.txt", 0, 2)

    def test_invalid_format(self) -> None:
        """Test invalid file format error."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("1 2\n3 4\n")  # Only 2 columns, but will test for 3
            fname = f.name
        try:
            with pytest.raises(ValueError, match="Invalid data format in"):
                validate_file_and_load(fname, 0, 3)
        finally:
            os.remove(fname)

    def test_valid_file(self) -> None:
        """Test successful file loading."""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            np.savetxt(f, arr)
            fname = f.name
        try:
            data = validate_file_and_load(fname, 0, 3)
            assert data.shape == arr.shape
        finally:
            os.remove(fname)


class TestGetMuAEpidermis:
    """Test cases for get_mu_a_epidermis function."""

    def test_file_is_found(self) -> None:
        interp, bounds = get_mu_a_epidermis()
        assert callable(interp)
        assert isinstance(bounds, tuple)
        assert bounds[0] < bounds[1]

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            get_mu_a_water("/not/a/real/file.txt")

    def test_valid_file(self) -> None:
        arr = np.array([[700, 0.1], [800, 0.2]])
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            # Write 5 header lines to skip
            f.write("# header\n" * 5)
            np.savetxt(f, arr)
            fname = f.name
        try:
            func, bounds = get_mu_a_epidermis(fname)
            # Check interpolation at a value in range
            x = arr[0, 0] * 1e-9
            assert bounds[0] <= x <= bounds[1]
            assert func(x) == pytest.approx(arr[0, 1] * 1e2)
        finally:
            os.remove(fname)


class TestGetMuAWater:
    """Test cases for get_mu_a_water function."""

    def test_file_is_found(self) -> None:
        interp, bounds = get_mu_a_water()
        assert callable(interp)
        assert isinstance(bounds, tuple)
        assert bounds[0] < bounds[1]

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            get_mu_a_water("/not/a/real/file.txt")

    def test_valid_file(self) -> None:
        arr = np.array([[700, 0.1], [800, 0.2]])
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            # Write 5 header lines to skip
            f.write("# header\n" * 5)
            np.savetxt(f, arr)
            fname = f.name
        try:
            func, bounds = get_mu_a_water(fname)
            # Check interpolation at a value in range
            x = arr[0, 0] * 1e-9
            assert bounds[0] <= x <= bounds[1]
            assert func(x) == pytest.approx(arr[0, 1])  # data already in m^-1
        finally:
            os.remove(fname)


class TestGetMuACollagen:
    """Test cases for get_mu_a_collagen function."""

    def test_file_is_found(self) -> None:
        interp, bounds = get_mu_a_collagen()
        assert callable(interp)
        assert isinstance(bounds, tuple)
        assert bounds[0] < bounds[1]

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            get_mu_a_collagen("/not/a/real/file.txt")

    def test_valid_file(self) -> None:
        arr = np.array([[700, 0.1], [800, 0.2]])
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            # Write 2 header lines to skip
            f.write("# header\n" * 2)
            np.savetxt(f, arr)
            fname = f.name
        try:
            func, bounds = get_mu_a_collagen(fname)
            x = arr[0, 0] * 1e-9
            assert bounds[0] <= x <= bounds[1]
            assert func(x) == pytest.approx(arr[0, 1] * 1e2)
        finally:
            os.remove(fname)


class TestGetOxidisedCytochromeC:
    """Test cases for get_oxidised_cytochrome_c function."""

    def test_file_is_found(self) -> None:
        interp, bounds = get_oxidised_cytochrome_c()
        assert callable(interp)
        assert isinstance(bounds, tuple)
        assert bounds[0] < bounds[1]

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            get_oxidised_cytochrome_c("/not/a/real/file.txt")

    def test_valid_file(self) -> None:
        arr = np.array([[600, 0.1], [800, 0.2], [960, 0.3]])
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            # Write 2 header lines to skip
            f.write("# header\n" * 2)
            np.savetxt(f, arr)
            fname = f.name
        try:
            func, bounds = get_oxidised_cytochrome_c(fname, smooth=False)
            x = arr[0, 0] * 1e-9
            assert bounds[0] <= x <= bounds[1]
            assert func(x) == pytest.approx(arr[0, 1] * 1e2)
        finally:
            os.remove(fname)


class TestGetReducedCytochromeC:
    """Test cases for get_reduced_cytochrome_c function."""

    def test_file_is_found(self) -> None:
        interp, bounds = get_reduced_cytochrome_c()
        assert callable(interp)
        assert isinstance(bounds, tuple)
        assert bounds[0] < bounds[1]

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            get_reduced_cytochrome_c("/not/a/real/file.txt")

    def test_valid_file(self) -> None:
        arr = np.array([[600, 0.1], [800, 0.2], [960, 0.3]])
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            # Write 2 header lines to skip
            f.write("# header\n" * 2)
            np.savetxt(f, arr)
            fname = f.name
        try:
            func, bounds = get_reduced_cytochrome_c(fname, smooth=False)
            x = arr[0, 0] * 1e-9
            assert bounds[0] <= x <= bounds[1]
            assert func(x) == pytest.approx(arr[0, 1] * 1e2)
        finally:
            os.remove(fname)


class TestGetBilirubinExtendedUpperRange:
    """Test cases for get_bilirubin_extended_upper_range function."""

    def test_file_is_found(self) -> None:
        interp, bounds = get_bilirubin_extended_upper_range()
        assert callable(interp)
        assert isinstance(bounds, tuple)
        assert bounds[0] < bounds[1]

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            get_bilirubin_extended_upper_range("/not/a/real/file.txt")

    def test_valid_file(self) -> None:
        arr = np.array([[600, 0.1], [800, 0.2]])
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            # Write 2 header lines to skip
            f.write("# header\n" * 2)
            np.savetxt(f, arr)
            fname = f.name
        try:
            func, bounds = get_bilirubin_extended_upper_range(fname, smooth=False)
            x = arr[0, 0] * 1e-9
            assert bounds[0] <= x <= bounds[1]
            # Bilirubin is abs()'d and *1e2
            assert func(x) == pytest.approx(abs(arr[0, 1]) * 1e2)
        finally:
            os.remove(fname)


class TestTransformationBase:
    """Test cases for base transformation class."""

    def setup_method(self) -> None:
        """Setup common parameters for tests."""
        self.wavelengths = torch.tensor([400.0, 500.0, 600.0])

    def test_transformation_init_valid(self) -> None:
        """Test valid transformation initialization."""
        # This should work for any transformation class
        transformation = Transformation(self.wavelengths)
        assert torch.all(transformation.wavelengths == self.wavelengths)

    def test_transformation_init_invalid_ndim(self) -> None:
        """Test error with invalid wavelength dimensions."""
        with pytest.raises(ValueError, match="Wavelengths must be a 1D tensor"):
            Transformation(self.wavelengths.clone()[None, :])

    def test_transformation_init_invalid_range(self) -> None:
        """Test error with wavelengths outside valid range."""
        wavelengths = self.wavelengths.clone()
        wavelengths[0] = 150.0  # Below 200 nm

        with pytest.raises(
            ValueError, match="Wavelengths must be between 200 and 2000 nm"
        ):
            Transformation(wavelengths)

        wavelengths = self.wavelengths.clone()
        wavelengths[0] = 2500.0  # Above 2000 nm

        with pytest.raises(
            ValueError, match="Wavelengths must be between 200 and 2000 nm"
        ):
            Transformation(wavelengths)

    def test_validate_tensor_range_valid(self) -> None:
        """Test valid tensor range validation."""
        tensor = torch.tensor([0.5, 0.7, 0.9])
        LanTransformation.validate_tensor_range(tensor, 0.0, 1.0, "test_tensor")

    def test_validate_tensor_range_invalid(self) -> None:
        """Test invalid tensor range validation."""
        tensor = torch.tensor([0.5, 1.5, 0.9])  # 1.5 > 1.0

        with pytest.raises(ValueError, match="test_tensor values must be in range"):
            LanTransformation.validate_tensor_range(tensor, 0.0, 1.0, "test_tensor")

        tensor = torch.tensor([-0.1, 0.5, 0.9])  # -0.1 < 0.0

        with pytest.raises(ValueError, match="test_tensor values must be in range"):
            LanTransformation.validate_tensor_range(tensor, 0.0, 1.0, "test_tensor")


class TestLanTransformation:
    """Test cases for Lan transformation."""

    def setup_method(self) -> None:
        """Setup common parameters for tests."""
        self.wavelengths = torch.tensor([450.0, 550.0, 650.0])
        self.sao2 = torch.tensor([0.8, 0.9])
        self.vhb = torch.tensor([0.02, 0.03])
        self.a_mie = torch.tensor([1000.0, 2000.0])
        self.b_mie = torch.tensor([1.5, 2.0])
        self.a_ray = torch.tensor([50.0, 100.0])
        self.g = torch.tensor([0.8, 0.9])

    def test_init(self) -> None:
        """Test Lan transformation initialization."""
        transformation = LanTransformation(self.wavelengths)

        assert transformation.wavelengths.shape == (len(self.wavelengths),)
        assert transformation.cHb == 150
        assert transformation.gmw_hb == 64500

    def test_absorption_valid(self) -> None:
        """Test valid absorption computation."""
        transformation = LanTransformation(self.wavelengths)
        result = transformation.absorption(self.sao2, self.vhb)

        assert result.shape == (2, len(self.wavelengths))
        assert torch.all(result >= 0)

    def test_absorption_invalid_sao2_range(self) -> None:
        """Test error with invalid sao2 range."""
        transformation = LanTransformation(self.wavelengths)
        sao2 = self.sao2.clone()
        sao2[0] = 1.5  # > 1.0

        with pytest.raises(ValueError, match="sao2"):
            transformation.absorption(sao2, self.vhb)

    def test_absorption_invalid_vhb_range(self) -> None:
        """Test error with invalid vhb range."""
        transformation = LanTransformation(self.wavelengths)
        vhb = self.vhb.clone()
        vhb[0] = 1.5  # > 1.0

        with pytest.raises(ValueError, match="vhb"):
            transformation.absorption(self.sao2, vhb)

    def test_absorption_length_mismatch(self) -> None:
        """Test error with length mismatch."""
        transformation = LanTransformation(self.wavelengths)

        with pytest.raises(ValueError, match="must have same length"):
            transformation.absorption(self.sao2, self.vhb[[0]])

    def test_scattering_valid(self) -> None:
        """Test valid scattering computation."""
        transformation = LanTransformation(self.wavelengths)

        result = transformation.scattering(self.a_mie, self.b_mie, self.a_ray, self.g)

        assert result.shape == (2, len(self.wavelengths))
        assert torch.all(result >= 0)

    def test_scattering_invalid_g_range(self) -> None:
        """Test error with invalid g range."""
        transformation = LanTransformation(self.wavelengths)
        g = self.g.clone()
        g[0] = 1.5  # > 1.0

        with pytest.raises(ValueError, match="anisotropy factor g"):
            transformation.scattering(self.a_mie, self.b_mie, self.a_ray, g)

        g = self.g.clone()
        g[0] = -0.1  # < 0.0

        with pytest.raises(ValueError, match="anisotropy factor g"):
            transformation.scattering(self.a_mie, self.b_mie, self.a_ray, g)


class TestTsuiTransformation:
    """Test cases for Tsui transformation."""

    def setup_method(self) -> None:
        """Setup common parameters for tests."""
        self.wavelengths = torch.tensor([460.0, 560.0, 660.0])
        self.sao2 = torch.tensor([0.8, 0.9])
        self.fhb = torch.tensor([0.02, 0.03])
        self.fw = torch.tensor([0.6, 0.7])
        self.fmel = torch.tensor([0.1, 0.2])
        self.a_mie = torch.tensor([1000.0, 2000.0])
        self.b_mie = torch.tensor([1.5, 2.0])
        self.a_ray = torch.tensor([50.0, 100.0])
        self.g = torch.tensor([0.8, 0.9])

    def test_init(self) -> None:
        """Test Tsui transformation initialization."""
        transformation = TsuiTransformation(self.wavelengths)

        assert transformation.wavelengths.shape == (len(self.wavelengths),)
        assert transformation.cHb == 150
        assert transformation.gmw_hb == 64500
        assert transformation.gmw_hbo2 == 64532

    def test_hemoglobin_total_absorption(self) -> None:
        """Test hemoglobin total absorption computation."""
        transformation = TsuiTransformation(self.wavelengths)

        result = transformation._hemoglobin_total_absorption(self.sao2)

        assert result.shape == (2, len(self.wavelengths))
        assert torch.all(result >= 0)

    def test_hemoglobin_total_absorption_invalid_sao2(self) -> None:
        """Test error with invalid sao2 range."""
        transformation = TsuiTransformation(self.wavelengths)
        sao2 = self.sao2.clone()
        sao2[0] = 1.5  # > 1.0

        with pytest.raises(ValueError, match="sao2"):
            transformation._hemoglobin_total_absorption(sao2)

    def test_mu_a_1_and_2(self) -> None:
        """Test mu_a_1_and_2 computation."""
        transformation = TsuiTransformation(self.wavelengths)

        result = transformation.mu_a_1_and_2()

        assert result.shape == (len(self.wavelengths),)
        assert torch.all(result >= 0)

    def test_mu_a_3(self) -> None:
        """Test mu_a_3 computation."""
        transformation = TsuiTransformation(self.wavelengths)

        result = transformation.mu_a_3(self.fmel)

        assert result.shape == (2, len(self.wavelengths))
        assert torch.all(result >= 0)

    def test_mu_a_3_invalid_fmel(self) -> None:
        """Test error with invalid fmel range."""
        transformation = TsuiTransformation(self.wavelengths)
        fmel = self.fmel.clone()
        fmel[0] = 1.1  # > 1.0

        with pytest.raises(ValueError, match="melanin volume fraction f_m"):
            transformation.mu_a_3(fmel)

    def test_mu_a_4(self) -> None:
        """Test mu_a_4 computation."""
        transformation = TsuiTransformation(self.wavelengths)

        result = transformation.mu_a_4(self.fhb, self.sao2, self.fw)

        assert result.shape == (2, len(self.wavelengths))
        assert torch.all(result >= 0)

    def test_mu_a_4_total_fraction_exceeds_one(self) -> None:
        """Test error when combined fractions exceed 1."""
        transformation = TsuiTransformation(self.wavelengths)
        fhb = self.fhb.clone()
        fhb[0] = 0.6

        with pytest.raises(ValueError, match="Combined hemoglobin and water fractions"):
            transformation.mu_a_4(fhb, self.sao2, self.fw)

    def test_scattering(self) -> None:
        """Test scattering computation."""
        transformation = TsuiTransformation(self.wavelengths)

        result = transformation.scattering(self.a_mie, self.b_mie, self.g)

        assert result.shape == (2, len(self.wavelengths))
        assert torch.all(result >= 0)

    def test_scattering_invalid_g_range(self) -> None:
        """Test error with invalid g range."""
        transformation = TsuiTransformation(self.wavelengths)
        g = self.g.clone()
        g[0] = 1.5  # > 1.0

        with pytest.raises(ValueError, match="anisotropy factor g"):
            transformation.scattering(self.a_mie, self.b_mie, g)

        g = self.g.clone()
        g[0] = 0.2  # < 0.5

        with pytest.raises(ValueError, match="anisotropy factor g"):
            transformation.scattering(self.a_mie, self.b_mie, g)


class TestManojlovicTransformation:
    """Test cases for Manojlovic transformation."""

    def setup_method(self) -> None:
        """Setup common parameters for tests."""
        self.wavelengths = torch.tensor([300.0, 500.0, 700.0])
        self.fhb = torch.tensor([0.02, 0.03])
        self.fhbo2 = torch.tensor([0.01, 0.02])
        self.fmel = torch.tensor([0.02, 0.03])
        self.fbrub = torch.tensor([0.01, 0.02])
        self.fcyto = torch.tensor([0.1, 0.2])
        self.fcyto_red = torch.tensor([0.1, 0.2])
        self.a = torch.tensor([2000.0, 5000.0])
        self.b_mie = torch.tensor([1.5, 2.0])
        self.f_ray = torch.tensor([1e-7, 2e-7])
        self.g = torch.tensor([0.8, 0.9])

    def test_init(self) -> None:
        """Test Manojlovic transformation initialization."""
        transformation = ManojlovicTransformation(self.wavelengths)

        assert transformation.wavelengths.shape == (len(self.wavelengths),)
        assert transformation.cHb == 150
        assert transformation.gmw_hb == 64500

    def test_melanin_absorption(self) -> None:
        """Test melanin absorption computation."""
        transformation = ManojlovicTransformation(self.wavelengths)

        result = transformation._melanin_absorption(self.fmel)

        assert result.shape == (2, len(self.wavelengths))
        assert torch.all(result >= 0)

    def test_melanin_absorption_invalid_range(self) -> None:
        """Test error with invalid melanin fraction range."""
        transformation = ManojlovicTransformation(self.wavelengths)

        f_m = torch.tensor([0.1])  # > 0.05

        with pytest.raises(ValueError, match="melanin volume fraction f_m"):
            transformation._melanin_absorption(f_m)

        f_m = torch.tensor([0.0])  # < 0.001

        with pytest.raises(ValueError, match="melanin volume fraction f_m"):
            transformation._melanin_absorption(f_m)

    def test_base_absorption(self) -> None:
        """Test base absorption computation."""
        transformation = ManojlovicTransformation(self.wavelengths)

        result = transformation._base_absorption()

        assert result.shape == (len(self.wavelengths),)
        assert torch.all(result >= 0)

    def test_hemoglobin_absorption(self) -> None:
        """Test hemoglobin absorption computation."""
        transformation = ManojlovicTransformation(self.wavelengths)

        result = transformation._hemoglobin_absorption(self.fhb)

        assert result.shape == (2, len(self.wavelengths))
        assert torch.all(result >= 0)

    def test_hemoglobin_absorption_invalid_range(self) -> None:
        """Test error with invalid hemoglobin fraction range."""
        transformation = ManojlovicTransformation(self.wavelengths)

        f_hb = torch.tensor([0.06])  # > 0.05

        with pytest.raises(ValueError, match="hemoglobin volume fraction f_hb"):
            transformation._hemoglobin_absorption(f_hb)

        f_hb = torch.tensor([0.0])  # < 0.001

        with pytest.raises(ValueError, match="hemoglobin volume fraction f_hb"):
            transformation._hemoglobin_absorption(f_hb)

    def test_oxyhemoglobin_absorption(self) -> None:
        """Test oxyhemoglobin absorption computation."""
        transformation = ManojlovicTransformation(self.wavelengths)

        result = transformation._oxyhemoglobin_absorption(self.fhbo2)

        assert result.shape == (2, len(self.wavelengths))
        assert torch.all(result >= 0)

    def test_oxyhemoglobin_absorption_invalid_range(self) -> None:
        """Test error with invalid oxyhemoglobin fraction range."""
        transformation = ManojlovicTransformation(self.wavelengths)

        f_hbo2 = torch.tensor([0.06])  # > 0.05

        with pytest.raises(ValueError, match="oxyhemoglobin volume fraction f_hbo2"):
            transformation._oxyhemoglobin_absorption(f_hbo2)

        f_hbo2 = torch.tensor([0.0])  # < 0.001

        with pytest.raises(ValueError, match="oxyhemoglobin volume fraction f_hbo2"):
            transformation._oxyhemoglobin_absorption(f_hbo2)

    def test_bilirubin_absorption(self) -> None:
        """Test bilirubin absorption computation."""
        transformation = ManojlovicTransformation(self.wavelengths)

        result = transformation._bilirubin_absorption(self.fbrub)

        assert result.shape == (2, len(self.wavelengths))
        assert torch.all(result >= 0)

    def test_bilirubin_absorption_invalid_range(self) -> None:
        """Test error with invalid bilirubin fraction range."""
        transformation = ManojlovicTransformation(self.wavelengths)

        f_brub = torch.tensor([0.11])  # > 0.1

        with pytest.raises(ValueError, match="bilirubin concentration f_brub"):
            transformation._bilirubin_absorption(f_brub)

        f_brub = torch.tensor([0.0])  # < 1e-7

        with pytest.raises(ValueError, match="bilirubin concentration f_brub"):
            transformation._bilirubin_absorption(f_brub)

    def test_cytochrome_c_ox_absorption(self) -> None:
        """Test oxidized cytochrome c absorption computation."""
        transformation = ManojlovicTransformation(self.wavelengths)

        result = transformation._cytochrome_c_ox_absorption(self.fcyto)

        assert result.shape == (2, len(self.wavelengths))
        assert torch.all(result >= 0)

    def test_cytochrome_c_ox_absorption_invalid_range(self) -> None:
        """Test error with invalid oxidized cytochrome c fraction range."""
        transformation = ManojlovicTransformation(self.wavelengths)

        f_cyto = torch.tensor([2.5])  # > 2

        with pytest.raises(
            ValueError, match="oxidised cytochrome c concentration f_cyto"
        ):
            transformation._cytochrome_c_ox_absorption(f_cyto)

        f_cyto = torch.tensor([0.0])  # < 1e-7

        with pytest.raises(
            ValueError, match="oxidised cytochrome c concentration f_cyto"
        ):
            transformation._cytochrome_c_ox_absorption(f_cyto)

    def test_cytochrome_c_red_absorption(self) -> None:
        """Test reduced cytochrome c absorption computation."""
        transformation = ManojlovicTransformation(self.wavelengths)

        result = transformation._cytochrome_c_red_absorption(self.fcyto_red)

        assert result.shape == (2, len(self.wavelengths))
        assert torch.all(result >= 0)

    def test_cytochrome_c_red_absorption_invalid_range(self) -> None:
        """Test error with invalid reduced cytochrome c fraction range."""
        transformation = ManojlovicTransformation(self.wavelengths)

        f_cyto_red = torch.tensor([2.5])  # > 2

        with pytest.raises(
            ValueError, match="reduced cytochrome c concentration f_cyto"
        ):
            transformation._cytochrome_c_red_absorption(f_cyto_red)

        f_cyto_red = torch.tensor([0.0])  # < 1e-7

        with pytest.raises(
            ValueError, match="reduced cytochrome c concentration f_cyto"
        ):
            transformation._cytochrome_c_red_absorption(f_cyto_red)

    def test_refractive_index(self) -> None:
        """Test refractive index computation."""
        transformation = ManojlovicTransformation(self.wavelengths)

        result = transformation.refractive_index()

        assert result.shape == (len(self.wavelengths),)
        assert torch.all(result > 1.0) and torch.all(result < 2.0)

    def test_anisotropy(self) -> None:
        """Test anisotropy computation."""
        transformation = ManojlovicTransformation(self.wavelengths)

        result = transformation.anisotropy()

        assert result.shape == (len(self.wavelengths),)
        assert torch.all(result >= 0) and torch.all(result <= 1)

    def test_mu_a_epi(self) -> None:
        """Test epidermis absorption computation."""
        transformation = ManojlovicTransformation(self.wavelengths)

        result = transformation.mu_a_epi(self.fmel)

        assert result.shape == (2, len(self.wavelengths))
        assert torch.all(result >= 0)

    def test_mu_a_dermis(self) -> None:
        """Test dermis absorption computation."""
        transformation = ManojlovicTransformation(self.wavelengths)

        result = transformation.mu_a_dermis(
            self.fhb,
            self.fhbo2,
            self.fbrub,
            self.fcyto_red,
            self.fcyto,
        )

        assert result.shape == (len(self.fhb), len(self.wavelengths))
        assert torch.all(result >= 0)

    def test_reduced_scattering(self) -> None:
        """Test reduced scattering computation."""
        transformation = ManojlovicTransformation(self.wavelengths)

        result = transformation._reduced_scattering(self.a, self.f_ray, self.b_mie)

        assert result.shape == (len(self.a), len(self.wavelengths))
        assert torch.all(result >= 0)

    def test_reduced_scattering_invalid_a_range(self) -> None:
        """Test error with invalid reduced scattering amplitude range."""
        transformation = ManojlovicTransformation(self.wavelengths)

        a = torch.tensor([10000.0])  # > 8000

        with pytest.raises(ValueError, match="scattering amplitude a"):
            transformation._reduced_scattering(a, self.f_ray, self.b_mie)

        a = torch.tensor([1000.0])  # < 2000

        with pytest.raises(ValueError, match="scattering amplitude a"):
            transformation._reduced_scattering(a, self.f_ray, self.b_mie)

    def test_reduced_scattering_invalid_f_ray_range(self) -> None:
        """Test error with invalid Rayleigh scattering fraction range."""
        transformation = ManojlovicTransformation(self.wavelengths)

        f_ray = torch.tensor([1e-5])  # > 1e-6

        with pytest.raises(ValueError, match="Rayleigh scattering fraction f_ray"):
            transformation._reduced_scattering(self.a, f_ray, self.b_mie)

        f_ray = torch.tensor([1e-9])  # < 1e-8

        with pytest.raises(ValueError, match="Rayleigh scattering fraction f_ray"):
            transformation._reduced_scattering(self.a, f_ray, self.b_mie)

    def test_scattering(self) -> None:
        """Test scattering computation."""
        transformation = ManojlovicTransformation(self.wavelengths)

        result = transformation.scattering(self.a, self.f_ray, self.b_mie)

        assert result.shape == (len(self.a), len(self.wavelengths))
        assert torch.all(result >= 0)


class TestConstants:
    """Test cases for constants classes."""

    def test_lan_constants(self) -> None:
        """Test Lan constants."""
        constants = LanConstants()

        assert constants.mu_a_range == (0, 1000)
        assert constants.mu_s_range == (10000, 35000)
        assert constants.g_range == (0.8, 0.9999)
        assert constants.n == 1.35
        assert constants.n_samples_sim == 5000
        assert constants.n_photons == 10**8
        assert constants.wavelengths.shape == (101,)
        assert constants.n_samples_surrogate == 100000
        assert constants.sao2_range == (0.001, 1.0)
        assert constants.vhb_range == (0.001, 1.0)
        assert constants.a_mie_range == (250, 6000)
        assert constants.b_mie_range == (0.1, 4.0)

    def test_manojlovic_constants(self) -> None:
        """Test Manojlovic constants."""
        constants = ManojlovicConstants()

        assert constants.f_mel_range == (0.001, 0.05)
        assert constants.f_hb_range == (0.001, 0.05)
        assert constants.f_hbo2_range == (0.001, 0.05)
        assert constants.f_brub_range == (10**-7, 0.1)
        assert constants.f_co_range == (10**-7, 2)
        assert constants.f_coo2_range == (10**-7, 2)
        assert constants.a_mie_range == (2000, 8000)
        assert constants.b_mie == 1.2
        assert constants.f_ray == 10**-7
        assert constants.d_epi == 0.0001
        assert constants.d_dermis == 0.01
        assert constants.n_samples_sim == 70000
        assert constants.n_photons == 10**6
        assert constants.wavelengths.shape == (351,)
        assert constants.n_samples_surrogate == 100000

    def test_tsui_constants(self) -> None:
        """Test Tsui constants."""
        constants = TsuiConstants()

        assert constants.mu_a_1_range == (10, 500)
        assert constants.mu_a_2_range == (10, 500)
        assert constants.mu_a_3_range == (100, 35000)
        assert constants.mu_a_4_range == (1, 1500)
        assert constants.mu_s_1_range == (10000, 100000)
        assert constants.mu_s_2_range == (1000, 50000)
        assert constants.mu_s_3_range == (1000 * 1.35, 50000 * 1.35)
        assert constants.mu_s_4_range == (1000, 50000)
        assert constants.f_w == 0.7
        assert constants.g_1 == 0.92
        assert constants.g_2 == 0.75
        assert constants.g_3 == 0.75
        assert constants.g_4 == 0.715
        assert constants.n == 1.42
        assert np.allclose(constants.d_1_range, (5e-6, 30e-6))
        assert np.allclose(constants.d_2_range, (5e-6, 30e-6))
        assert np.allclose(constants.d_3_range, (10e-6, 60e-6))
        assert constants.n_samples_sim == 30000
        assert constants.n_photons == 10**8
        assert constants.wavelengths.shape == (151,)
        assert constants.n_samples_surrogate == 100000
        assert np.allclose(constants.a_1_range, (1e7, 5e8))
        assert np.allclose(constants.a_2_range, (1e7, 5e8))
        assert np.allclose(constants.a_4_range, (1e7, 5e8))
        assert constants.b_1_range == (1, 2)
        assert constants.b_2_range == (1, 2)
        assert constants.b_4_range == (1, 2)
        assert constants.f_m_range == (0.01, 0.25)
        assert constants.f_b_range == (0.0, 0.005)
        assert constants.sao2_range == (0.0, 1.0)
