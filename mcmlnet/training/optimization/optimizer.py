"""Train surrogate model with Pytorch Lightning"""

import os

import hydra
import lightning as pl
import omegaconf
from dotenv import load_dotenv
from hydra.utils import instantiate
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from omegaconf import DictConfig, OmegaConf, open_dict

from mcmlnet.training.flop_and_gmac_counting import FLOPAndMACCounter
from mcmlnet.utils.logging import setup_logging

load_dotenv()
logger = setup_logging(level="info", logger_name=__name__)


def train_loop(cfg: DictConfig, seed: int = 42) -> None:
    """Train loop for Pytorch Lightning

    Args:
        cfg: Configuration for the training process
        seed: Random, global seed for reproducibility (default: 42)

    Raises:
        omegaconf.errors.ConfigAttributeError: If configuration is missing
            required attributes
        AttributeError: If model does not have a generator attribute
    """
    logger.info(f"Training with following config:\n{OmegaConf.to_yaml(cfg)}")
    pl.seed_everything(seed)

    # Load data module
    data_module = instantiate(cfg.data_module, cfg=cfg)
    cfg = data_module.cfg

    # Save every two validation steps
    callback = ModelCheckpoint(
        monitor="val_loss",
        filename=f"{cfg.name}" + "-{epoch}-{val_loss:.4f}",
        every_n_epochs=2 * cfg.trainer.check_val_every_n_epoch,
    )
    # Progress bar callback
    bar = TQDMProgressBar(refresh_rate=cfg.progress_bar_refresh_rate)
    # Early stopping callback
    early = EarlyStopping(monitor="val_loss", mode="min")
    # Instantiate profiler
    profiler = instantiate(cfg.profiler)
    # Instantiate trainer
    trainer = pl.Trainer(
        **cfg.trainer,
        profiler=profiler,
        callbacks=[callback, bar, early],
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

    # compute GFLOPs and GMACs per epoch
    compute = FLOPAndMACCounter(model)
    gflops, gmacs = compute(
        data_module.train_dataloader(), n_trainable_params, per_epoch=False
    )
    with open_dict(cfg):
        cfg.gflop_per_epoch = gflops
        cfg.gmac_per_epoch = gmacs

    # save OmegaConf in PyTorch Lightning/ Tensorboard structure
    save_dir = trainer.logger.log_dir
    logger.info(f"Save dir set to: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(config=cfg, f=save_dir + "/config_log.yaml")

    # Training and evaluation
    logger.info(f"Training path set to: {cfg.ckpt_path}")
    trainer.fit(model, datamodule=data_module, ckpt_path=cfg.ckpt_path)
    trainer.test(ckpt_path="best", datamodule=data_module)


@hydra.main(  # type: ignore [misc]
    config_path=os.getenv("CFG_DIR"), config_name="default", version_base="1.3.2"
)
def main(cfg: DictConfig) -> None:
    """Instantiate datasets, model and Trainer and concurrent training"""
    try:
        if isinstance(cfg.preprocessing.kfolds, int):
            # Iterate over kfold splits
            for fold in range(cfg.preprocessing.kfolds):
                # Update which fold to run
                if fold != 0:
                    continue
                cfg.preprocessing.fold = fold

                # Remove potential previous transformations
                if "norm_1" in cfg.preprocessing:
                    del cfg.preprocessing.norm_1
                if "norm_2" in cfg.preprocessing:
                    del cfg.preprocessing.norm_2
                if "pca_transformation_list" in cfg.preprocessing:
                    del cfg.preprocessing.pca_transformation_list
                if "pca_mean_list" in cfg.preprocessing:
                    del cfg.preprocessing.pca_mean_list
                train_loop(cfg)
        else:
            train_loop(cfg)
    except omegaconf.errors.ConfigAttributeError:
        train_loop(cfg)


if __name__ == "__main__":
    main()
