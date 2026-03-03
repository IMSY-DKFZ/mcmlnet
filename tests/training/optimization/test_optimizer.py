"""Tests for mcmlnet.training.optimization.optimizer module."""

import os

import pytest
import torch
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir

from mcmlnet.training.optimization.optimizer import main

load_dotenv()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")  # type: ignore[misc]
def test_train_loop_short_run() -> None:
    """Test run train_loop with a minimal configuration for a very short run."""
    with initialize_config_dir(config_dir=os.environ["CFG_DIR"], version_base="1.3.2"):
        cfg = compose(config_name="default")
        cfg.trainer.max_epochs = 5
        cfg.trainer.check_val_every_n_epoch = 1
        cfg.model.generator.layers = [10, 10, 10, 1]

    main(cfg)
