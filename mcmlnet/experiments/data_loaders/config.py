"""Configuration and constants for data loading operations."""

from typing import ClassVar


class DataConfig:
    """Configuration class containing all data loading constants and mappings."""

    # Supported simulation types
    SUPPORTED_SIMULATIONS: ClassVar[list[str]] = [
        "generic_sims",
        "lan_sims",
        "tsui_sims",
        "manoj_sims",
        "jacques_sims",
        "jacques_sims_artificial",
    ]

    # Supported ablation types
    SUPPORTED_ABLATIONS: ClassVar[list[str]] = [
        "base_subset",
        "discrete_g_subset",
        "restricted_vhb_subset",
        "single_g_subset",
        "single_layer_subset",
        "single_n_subset",
        "superset",
        "single_layer_superset",
    ]

    # Simulation dataset names and display labels
    SIMULATION_NAMES: ClassVar[dict[str, str]] = {
        "generic_sims": "Our Simulations",
        "lan_sims": "Lan et al. (2023)",
        "tsui_sims": "Tsui et al. (2018)",
        "manoj_sims": "Manojlovic et al. (2025)",
        "jacques_sims": "Jacques et al. (1999)",
        "jacques_sims_artificial": "Jacques et al. (1999) (increased vHb)",
        "generic_sims_8400k": "Our Simulations (8400k Training Samples)",
        "generic_sims_420k": "Our Simulations (420k Training Samples)",
        "generic_sims_21k": "Our Simulations (21k Training Samples)",
    }

    # Ablation study dataset names and display labels
    ABLATION_NAMES: ClassVar[dict[str, str]] = {
        "base_subset": "Ablation Base Data",
        "discrete_g_subset": r"Ablation Discrete $g$",
        "restricted_vhb_subset": r"Ablation Restricted $v_{Hb}$",
        "single_g_subset": r"Ablation Single $g$",
        "single_layer_subset": "Ablation Single Layer",
        "single_n_subset": r"Ablation Single $n$",
        "superset": "Ablation Hemoglobin Superset",
        "single_layer_superset": "Ablation Single Layer Hem. Superset",
    }

    # Real dataset names and display labels
    REAL_NAMES: ClassVar[dict[str, str]] = {
        "pig_semantic": "Pig Semantic",
        "human": "Human",
        "pig_masks": "Pig Masks",
    }

    # Common defaults
    DEFAULT_SUBSAMPLE_SIZE_TISSUE_MODEL: int = 70000
    DEFAULT_RANDOM_SEED: int = 42
    DEFAULT_VAL_PERCENT: float = 0.1
    DEFAULT_TEST_PERCENT: float = 0.2
    DEFAULT_N_WAVELENGTHS: int = 100
    DEFAULT_N_LAYERS: int = 3

    # Surrogate model defaults
    DEFAULT_SUBSAMPLE_SIZE_SURROGATE_MODEL: int = 100000
    JACQUES_DEFAULT_REFRACTIVE_INDEX: float = 1.44

    # Data split names
    DATA_SPLIT_NAMES: ClassVar[list[str]] = ["train", "val", "test"]
    SUPPORTED_DATASETS: ClassVar[list[str]] = ["pig_semantic", "human", "pig_masks"]
