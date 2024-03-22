from typing import Any, Callable, List, Optional, List, Sequence
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, LightningModule
import warnings
import pytorch_lightning as pl
import torch.nn
from pytorch_lightning.utilities import rank_zero_only
import time
from omegaconf import OmegaConf
from multimodal_contrastive.networks.components import MultiTask_model, JointEncoder
import logging

log = logging.getLogger()
log.setLevel(logging.INFO)


def instantiate_model(cfg: DictConfig):
    """Instantiates PL module from config.

    Parameters:
    cfg (DictConfig): config

    Returns:
    model (LightningModule)
    """
    if cfg.model._target_ == "multimodal_contrastive.networks.models.GMC_PL":
        # Init encoder_joint
        logging.info(f"Instantiating torch.nn.module JointEncoder")
        encoder_joint: torch.nn.module = JointEncoder(
            encoders_mod=hydra.utils.instantiate(cfg.model.encoders_mod)
        )

        # Init lightning model
        logging.info(f"Instantiating lightning model <{cfg.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(
            cfg.model,
            encoder_joint=encoder_joint,
        )
    elif cfg.model._target_ == "multimodal_contrastive.networks.models.CLIP_PL":
        # Init lightning model
        logging.info(f"Instantiating lightning model <{cfg.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(cfg.model)

    return model


def instantiate_callbacks(cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config.

    Parameters:
    cfg (DictConfig): config.

    Returns:
    List[Callback]: List with all instantiated callbacks.
    """

    callbacks_cfg = cfg.get("callbacks")
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        print(callbacks_cfg)
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")

            cb_instance = hydra.utils.instantiate(cb_conf)
            callbacks.append(cb_instance)

    return callbacks


def instantiate_evaluations(cfg: DictConfig) -> List[Callback]:
    """Instantiates evaluation callbacks from config.

    Parameters:
    cfg (DictConfig): config.

    Returns:
    List[Callback]: List with all instantiated callbacks.
    """

    callbacks_cfg = cfg.get("evaluations")
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if cb_conf._target_ in [
            "multimodal_contrastive.evaluation.evaluation.LinearProbeOnlineEvaluator",
            "multimodal_contrastive.evaluation.evaluation.LinearProbeFinalEvaluator",
        ]:
            backbone = instantiate_model(cfg)
            lp_model = MultiTask_model(
                num_tasks=1310,
                loss_name="mtl_bceloss",
                backbone=backbone,
                mod_name="struct",
                freeze_backbone=False,
            )
            cb_instance = hydra.utils.instantiate(cb_conf, model=lp_model)

        elif (
            cb_conf._target_
            == "multimodal_contrastive.evaluation.evaluation.RetrievalOnlineEvaluator"
        ):
            model = instantiate_model(cfg)
            cb_instance = hydra.utils.instantiate(cb_conf, model=model)

        elif (
            cb_conf._target_
            == "multimodal_contrastive.evaluation.evaluation.LatentDistCorrelationEvaluator"
        ):
            model = instantiate_model(cfg)
            cb_instance = hydra.utils.instantiate(cb_conf, model=model)

        else:
            cb_instance = hydra.utils.instantiate(cb_conf)

        callbacks.append(cb_instance)

    return callbacks


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.
    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    dict_model = OmegaConf.to_container(cfg.model)
    if dict_model.get("encoders_mod") is not None:
        for ix, module_mod in enumerate(dict_model["encoders_mod"]):
            hparams["model/encoders_mod/{}".format(ix)] = module_mod
    if dict_model.get("projectors_mod") is not None:
        for ix, module_mod in enumerate(dict_model["projectors_mod"]):
            hparams["model/projectors_mod/{}".format(ix)] = module_mod

    # save others
    hparams["datamodule"] = OmegaConf.to_container(cfg.datamodule)
    hparams["trainer"] = OmegaConf.to_container(cfg.trainer)
    hparams["ckp_path"] = trainer.checkpoint_callback.best_model_path

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)
