"""
Hyperparameter tuning script using Optuna.

"""

import os
from datetime import datetime
from typing import Any

import hydra
import lightning as pl
import optuna
from dotenv import load_dotenv
from hydra.utils import instantiate
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf, open_dict
from optuna.integration import PyTorchLightningPruningCallback

from mcmlnet.training.flop_and_gmac_counting import FLOPAndMACCounter
from mcmlnet.utils.logging import setup_logging

load_dotenv()
logger = setup_logging(level="info", logger_name=__name__)


# Constants
DEFAULT_BASE_DATASET_SIZE = 640000
DEFAULT_TARGET_STEPS = 2000000
MIN_VAL_CHECK_EPOCHS = 5
MAX_VAL_CHECK_EPOCHS = 25


def _calculate_derived_parameters(
    cfg: DictConfig, max_augm_epochs_ratio: float, patience_ratio: float
) -> dict[str, int]:
    """Calculate derived training parameters."""
    base_approx_dataset_size = DEFAULT_BASE_DATASET_SIZE * cfg.train_data_ratio
    batches_per_epoch = base_approx_dataset_size // cfg.model.batch_size
    max_epochs = int(DEFAULT_TARGET_STEPS / batches_per_epoch)
    max_augm_epochs = int(max_epochs * max_augm_epochs_ratio)
    scheduler_patience = int(max_augm_epochs * patience_ratio)
    val_check_frequency = max(
        MIN_VAL_CHECK_EPOCHS, min(MAX_VAL_CHECK_EPOCHS, max_epochs // 50)
    )

    return {
        "batches_per_epoch": batches_per_epoch,
        "max_epochs": max_epochs,
        "max_augm_epochs": max_augm_epochs,
        "scheduler_patience": scheduler_patience,
        "val_check_frequency": val_check_frequency,
    }


def _update_config_with_derived_params(
    cfg: DictConfig, derived: dict, noise_snr: float
) -> None:
    """Update configuration with derived parameters."""
    # Update training parameters
    cfg.trainer.max_epochs = derived["max_epochs"]
    cfg.trainer.check_val_every_n_epoch = derived["val_check_frequency"]
    cfg.model.scheduler_gen.patience = derived["scheduler_patience"]

    # Update augmentation parameters
    for aug_type in ["input_augmentations", "output_augmentations"]:
        cfg[aug_type].noise.batches_per_epoch = derived["batches_per_epoch"]
        cfg[aug_type].noise.max_epochs = derived["max_augm_epochs"]
        cfg[aug_type].noise.snr = noise_snr

    # Set consistent max_augm_epochs
    cfg.max_augm_epochs = derived["max_augm_epochs"]
    cfg.model.max_augm_epochs = derived["max_augm_epochs"]


def _suggest_hyperparameters(
    trial: optuna.Trial, cfg: DictConfig
) -> tuple[float, float, float, int, int]:
    """Suggest hyperparameters from trial and update config."""
    # Training hyperparameters
    max_augm_epochs_ratio = trial.suggest_categorical(
        "max_augm_epochs_ratio", cfg.tune_config.max_augm_epochs_ratio
    )
    patience_ratio = trial.suggest_categorical(
        "patience_ratio", cfg.tune_config.patience_ratio
    )
    cfg.model.batch_size = trial.suggest_categorical(
        "batch_size", cfg.tune_config.batch_size
    )
    noise_snr = trial.suggest_categorical("snr", cfg.tune_config.snr)

    # Generator hyperparameters
    generator_depth = trial.suggest_int(
        "depth", cfg.tune_config.gen_depth.low, cfg.tune_config.gen_depth.high
    )
    layer_width = trial.suggest_categorical(
        "layer_width", cfg.tune_config.gen_layer_width
    )

    # Update generator architecture
    if "kan" in str(cfg.name).lower():
        cfg.model.generator.layers_hidden = [15] + [layer_width] * generator_depth + [1]
        cfg.model.generator.grid_size = trial.suggest_int(
            "grid_size",
            cfg.tune_config.gen_grid_size.low,
            cfg.tune_config.gen_grid_size.high,
        )
        grid_range = trial.suggest_float(
            "grid_range",
            cfg.tune_config.gen_grid_range.low,
            cfg.tune_config.gen_grid_range.high,
        )
        cfg.model.generator.grid_range = [-grid_range, grid_range]
        cfg.model.optimizer_gen.rho = trial.suggest_float(
            "rho", cfg.tune_config.rho.low, cfg.tune_config.rho.high
        )
    else:
        cfg.model.generator.layers = [layer_width] * generator_depth + [1]

    # Optimizer hyperparameters
    cfg.model.optimizer_gen.lr = trial.suggest_float(
        "lr", cfg.tune_config.gen_lr.low, cfg.tune_config.gen_lr.high, log=True
    )
    cfg.model.optimizer_gen.weight_decay = trial.suggest_float(
        "weight_decay",
        cfg.tune_config.weight_decay.low,
        cfg.tune_config.weight_decay.high,
        log=True,
    )

    return (
        max_augm_epochs_ratio,
        patience_ratio,
        noise_snr,
        generator_depth,
        layer_width,
    )


def train_scaling(trial: optuna.Trial, cfg: DictConfig, seed: int = 42) -> float:
    """Train model with parameters from Optuna trial and return validation loss.

    Args:
        trial: Optuna trial object containing hyperparameters
        cfg: Configuration DictConfig from hydra
        seed: Random seed for reproducibility (default: 42)

    Returns:
        float: The validation loss
    """
    # Update config with trial suggestions
    with open_dict(cfg):
        (
            max_augm_epochs_ratio,
            patience_ratio,
            noise_snr,
            generator_depth,
            layer_width,
        ) = _suggest_hyperparameters(trial, cfg)

        # Calculate dependent parameters and update config
        derived = _calculate_derived_parameters(
            cfg, max_augm_epochs_ratio, patience_ratio
        )
        _update_config_with_derived_params(cfg, derived, noise_snr)

        # Log trial parameters
        trial_params = {
            "max_augm_epochs_ratio": max_augm_epochs_ratio,
            "patience_ratio": patience_ratio,
            "batch_size": cfg.model.batch_size,
            "noise_snr": noise_snr,
            "generator_depth": generator_depth,
            "layer_width": layer_width,
            "learning_rate": cfg.model.optimizer_gen.lr,
            "weight_decay": cfg.model.optimizer_gen.weight_decay,
            **{
                k: v for k, v in derived.items() if k != "batches_per_epoch"
            },  # Skip internal calculation
        }
        logger.info(f"Trial #{trial.number} parameters: %s", trial_params)

    # Set the seed for reproducibility
    pl.seed_everything(seed)

    # Load data module
    data_module = instantiate(cfg.data_module, cfg=cfg)

    # Create a unique directory for this trial
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_dir = os.path.join(
        os.getcwd(), f"optuna_trials/trial_{trial.number}_{timestamp}"
    )
    os.makedirs(trial_dir, exist_ok=False)

    # Setup callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=1,  # reduce patience to 1 validation cycle for more aggressive pruning
        min_delta=1e-5,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(trial_dir, "checkpoints"),
        filename="best_model_{epoch:02d}_{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    # Workaround for pytorch lightning 2.0 https://github.com/optuna/optuna/issues/4689
    class OptunaPruning(PyTorchLightningPruningCallback, pl.Callback):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)

    optuna_callback = OptunaPruning(trial, monitor="val_loss")

    # Setup logger
    tb_logger = TensorBoardLogger(save_dir=trial_dir, name="", version=".")
    # Setup trainer
    trainer = pl.Trainer(
        **cfg["trainer"],
        enable_checkpointing=True,
        logger=tb_logger,
        callbacks=[
            early_stopping,
            checkpoint_callback,
            optuna_callback,
        ],
    )

    # Instantiate model
    with trainer.init_module():
        model = instantiate(cfg.model, cfg=cfg, data_module=data_module)

    # Log amount of trainable network parameters for easier evaluation
    with open_dict(cfg):
        try:
            n_trainable_params = sum(
                p.numel() for p in model.generator.parameters() if p.requires_grad
            )
        except AttributeError:
            n_trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
        logger.info(f"Number of trainable parameters: {n_trainable_params}")
        cfg.n_trainable_params = n_trainable_params

    # Compute GFLOPs and GMACs per epoch
    compute = FLOPAndMACCounter(model)
    gflops, gmacs = compute(
        data_module.train_dataloader(), n_trainable_params, per_epoch=False
    )
    with open_dict(cfg):
        cfg.gflop_per_epoch = gflops
        cfg.gmac_per_epoch = gmacs

    # Save config for this trial
    OmegaConf.save(config=cfg, f=os.path.join(trial_dir, "config_log.yaml"))

    # Train model
    trainer.fit(model, datamodule=data_module)

    return trainer.callback_metrics["val_loss"].item()  # type: ignore [no-any-return]


@hydra.main(  # type: ignore [misc]
    config_path=os.getenv("CFG_DIR"), config_name="default", version_base="1.3.2"
)
def main(cfg: DictConfig) -> None:
    logger.info(f"Tuning with following config:\n{OmegaConf.to_yaml(cfg)}")

    # Create study name and storage
    study_name = f"optuna_scaling_sigmoid_KAN_{str(cfg.train_data_ratio).replace('.', '_')}_data_ratio"  # noqa: E501
    storage_name = f"sqlite:///{study_name}.db"

    # Create or load study
    if cfg.get("resume", False) and os.path.exists(f"{study_name}.db"):
        logger.info(f"Loading existing study: {study_name}")
        study = optuna.load_study(study_name=study_name, storage=storage_name)
    else:
        logger.info(f"Creating new study: {study_name}")
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction="minimize",
            pruner=optuna.pruners.HyperbandPruner(min_resource=5),
        )

    # Run optimization
    study.optimize(lambda trial: train_scaling(trial, cfg), n_trials=cfg.num_trials)

    # Report results
    trials = study.trials
    logger.info("Optimization finished.")
    logger.info(f"Total trials: {len(trials)}")
    logger.info(
        "Pruned trials: "
        f"{sum(1 for t in trials if t.state == optuna.trial.TrialState.PRUNED)}"
    )
    logger.info(
        "Complete trials: "
        f"{sum(1 for t in trials if t.state == optuna.trial.TrialState.COMPLETE)}"
    )
    logger.info(
        f"Best trial - Value: {study.best_trial.value:.6f}, "
        f"Params: {study.best_trial.params}"
    )


if __name__ == "__main__":
    main()
