import numpy as np
import pandas as pd
import pytest

from mcmlnet.utils.load_configs import (
    get_split_name,
    label_is_organic,
    load_config,
    load_label_map,
)


class TestLoadConfig:
    def test_load_config(self) -> None:
        """Test loading configuration file."""
        cfg = load_config()
        assert isinstance(cfg, dict)
        assert all(isinstance(key, str) for key in cfg.keys())
        assert "data" in cfg
        assert "surrogate" in cfg
        assert "pca" in cfg
        assert "recall" in cfg


class TestLabelMap:
    """Test loading label maps and related functions."""

    def test_load_label_map_valid(self) -> None:
        """Test loading valid label map for human dataset."""
        label_map = load_label_map("human")
        assert isinstance(label_map, dict)
        assert label_map[0] == "artery"
        assert label_map[6] == "cauterized tissue"
        assert label_map[11] == "fat (subcutaneous)"
        assert label_map[17] == "hepatic ligament"
        assert label_map[255] == "unlabeled"

    def test_load_label_map_invalid(self) -> None:
        """Test loading label map for invalid dataset."""
        with pytest.raises(ValueError):
            load_label_map("pig")


class TestLabelIsOrganic:
    """Test filtering organic labels."""

    def setup_method(self) -> None:
        self.example_labels = [0, 1, 2, 3, 4, 255]
        self.example_is_organic = [True, False, True, True, False, False]

    def test_label_is_organic_list(self) -> None:
        mask = label_is_organic(self.example_labels, "human")
        assert isinstance(mask, np.ndarray)
        assert np.array_equal(mask, self.example_is_organic)

    def test_label_is_organic_ndarray(self) -> None:
        mask = label_is_organic(np.array(self.example_labels), "human")
        assert isinstance(mask, np.ndarray)
        assert np.array_equal(mask, self.example_is_organic)

    def test_label_is_organic_series(self) -> None:
        mask = label_is_organic(pd.Series(self.example_labels), "human")
        assert isinstance(mask, np.ndarray)
        assert np.array_equal(mask, self.example_is_organic)

    def test_label_is_organic_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unknown dataset"):
            label_is_organic(self.example_labels, "pig")


class TestGetSplitName:
    """Test getting split names for datasets."""

    def test_get_split_name_valid(self) -> None:
        assert get_split_name("pig_masks") == "fold_0"
        assert get_split_name("pig_semantic") == "fold_P044,P050,P059"
        assert get_split_name("human") == "fold_0"

    def test_get_split_name_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_split_name("unknown")
