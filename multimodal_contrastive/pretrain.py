import numpy as np
from typing import Dict, Union, List
import hydra
from omegaconf import OmegaConf
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from multimodal_contrastive.utils import utils

import logging

logging.getLogger().setLevel(logging.INFO)
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)


OmegaConf.register_new_resolver("sum", lambda input_list: np.sum(input_list), replace=True)


@hydra.main(config_path="../configs", config_name="puma_smg_gmc", version_base=None)
def train(cfg):
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        logging.info(f"Seed everything with <{cfg.seed}>")
        seed_everything(cfg.seed, workers=True)

    # Print configuration
    print(cfg)

    # Init lightning datamodule
    logging.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.datamodule, seed=cfg.seed
    )

    # Init lightning module
    model: LightningModule = utils.instantiate_model(cfg)

    # Init callbacks
    logging.info(f"Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg)

    # Init evaluator
    logging.info(f"Instantiating evaluations...")
    evaluations: List[Callback] = utils.instantiate_evaluations(cfg)

    # Init lightning trainer
    logging.info(f"Instantiating trainer... <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks + evaluations
    )

    trainer.fit(model, datamodule, ckpt_path=cfg.get("ckpt_path"))
    trainer.test(model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"), verbose=True)

    print("Best ckp path: {}".format(trainer.checkpoint_callback.best_model_path))

    # Send some parameters from config to all lightning loggers
    log_dict = {
        "cfg": cfg,
        "model": model,
        "trainer": trainer,
    }
    logging.info(f"Logging hyperparameters!")
    utils.log_hyperparameters(log_dict)


if __name__ == "__main__":
    train()
