from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from omegaconf import ListConfig, OmegaConf, DictConfig
from multimodal_contrastive.networks.utils import move_batch_input_to_device
from typing import Iterable, List, Optional, Set, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import ModuleList, ModuleDict
import pytorch_lightning as pl
from pytorch_lightning import LightningModule

from ..networks.loss import get_loss, MultiTaskLoss
from ..networks.components import (
    MultiLayerPerceptron,
    CommonEncoder,
    JointEncoder,
    MultiTask_model,
)


class MultiModalContrastive_PL(pl.LightningModule):
    """Parent class for triple and dual contrastive learning module.

    Parameters
    ----------
    loss_name (string) : name of the loss
    projectors_mod (List[torch.nn.Module]): list of projector modules for each modality
    encoders_mod (List[torch.nn.Module]): list of encoder modules for each modality
    temperature (float): temperature parameter for calculate contrastive loss
    lr (float): learning rate
    """

    def __init__(
        self,
        loss_name,
        encoders_mod,
        projectors_mod,
        temperature,
        lr,
    ):
        super().__init__()

        self.temperature = temperature
        self.lr = lr
        self.loss = get_loss(name=loss_name)
        if isinstance(encoders_mod, DictConfig):
            encoders_mod = OmegaConf.to_object(encoders_mod)
        if isinstance(projectors_mod, DictConfig):
            projectors_mod = OmegaConf.to_object(projectors_mod)
        self.encoders = ModuleDict()
        self.encoders.update(encoders_mod)
        self.projectors = ModuleDict()
        self.projectors.update(projectors_mod)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def _mod_encode(self, x_mod, encoder_mod):
        raise NotImplementedError

    def compute_representations(self, x_dict, mod_name=None):
        if mod_name is not None:
            return self._mod_encode(
                x_dict[mod_name], self.encoders[mod_name], self.projectors[mod_name]
            )

        else:
            batch_representations = {}
            for mod_name, x in x_dict.items():
                mod_representations = self._mod_encode(
                    x_dict[mod_name], self.encoders[mod_name], self.projectors[mod_name]
                )
                batch_representations[mod_name] = mod_representations
            return batch_representations

    def forward(self, batch, mod_name=None):
        x_dict = batch["inputs"]
        return self.compute_representations(x_dict, mod_name=mod_name)

    def compute_representation_dataloader(
        self,
        dataloader,
        device=None,
        mod_name=None,
        disable_progress_bar=False,
        return_mol=False,
    ):
        outs = defaultdict(list)
        returned_mols = []
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for batch in tqdm(
                dataloader, position=0, leave=True, disable=disable_progress_bar
            ):
                x_dict = batch["inputs"]
                if return_mol:
                    returned_mols.extend([*x_dict["struct"].mols])
                x_dict = move_batch_input_to_device(x_dict, device=device)
                
                batch_representations = self.compute_representations(
                    x_dict, mod_name=mod_name
                )
                for k, v in batch_representations.items():
                    outs[k].append(v.cpu().data.numpy())

        final_outs = dict()
        for mod in outs.keys():
            final_outs[mod] = np.vstack(outs[mod])

        if return_mol:
            return final_outs, returned_mols
        else:
            return final_outs

    def _step(self, batch, batch_idx, step_name, dataloader_idx=None):
        batch_size = batch["labels"].shape[0]
        batch_representations = self.forward(batch)
        loss = self.loss(batch_representations, self.temperature, batch_size)

        return loss

    def training_step(self, train_batch, batch_idx):
        loss = self._step(batch=train_batch, batch_idx=batch_idx, step_name="train")

        batch_size = train_batch["labels"].shape[0]

        self.log(
            "train/loss",
            loss,
            sync_dist=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )

        return loss

    def validation_step(self, val_batch, batch_idx, dataloader_idx=None):
        loss = self._step(
            batch=val_batch,
            batch_idx=batch_idx,
            step_name="val",
            dataloader_idx=dataloader_idx,
        )

        batch_size = val_batch["labels"].shape[0]

        self.log(
            "val/loss",
            loss,
            sync_dist=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )

        return loss


class GMC_PL(MultiModalContrastive_PL):
    """Triple contrastive learning module (for gmc with puma dataset).

    Parameters
    ----------
    loss_name (string) : name of the loss
    encoders_mod (List[torch.nn.Module]): list of modules for each modality as the component of the model
    encoders_joint (torch.nn.Module]): JointEncoder module that takes in all modality input to create joint representations
    projectors_mod (List[torch.nn.Module]): list of modules for each modality as the components of the model
    common_encoder (torch.nn.Module): component of the GMC PL
    temperature (float): temperature for the loss term
    lr (float): learning rate
    """

    def __init__(
        self,
        loss_name,
        encoders_mod,
        encoder_joint,
        projectors_mod,
        common_encoder,
        temperature,
        lr,
        **kwargs,
    ):
        super().__init__(loss_name=loss_name, encoders_mod=encoders_mod, projectors_mod=projectors_mod, temperature=temperature, lr=lr)
        self.save_hyperparameters()
        self.encoder_joint = encoder_joint
        self.encoders.update({"joint": self.encoder_joint})
        self.common_encoder = common_encoder
        

    def _mod_encode(self, x_mod, encoders_mod, proj):
        emb_mod_ = encoders_mod(x_mod)
        emb_mod = proj(emb_mod_)
        mod_representations = self.common_encoder(emb_mod)
        return mod_representations



class CLIP_PL(MultiModalContrastive_PL):
    """Dual contrastive learning module (for mocop with jump dataset).

    Parameters
    ----------
    loss_name (string) : name of the loss
    encoders_mod (List[torch.nn.Module]): list of modules for each modality as the component of the model
    projectors_mod (List[torch.nn.Module]): list of modules for each modality as the components of the model
    temperature (float): temperature for the loss term
    lr (float): learning rate
    """

    def __init__(
        self,
        loss_name,
        encoders_mod,
        projectors_mod,
        temperature,
        lr,
        **kwargs,
    ):
        super().__init__(loss_name=loss_name, encoders_mod=encoders_mod, projectors_mod=projectors_mod, temperature=temperature, lr=lr)
        self.save_hyperparameters()

    def _mod_encode(self, x_mod, encoders_mod, proj):
        emb_mod_ = encoders_mod(x_mod)
        emb_mod = proj(emb_mod_)
        return emb_mod


class MultiTask_PL(pl.LightningModule):
    def __init__(self, loss_name, num_tasks, lr, backbone=None, ckp=None, backbone_name=None, mod_name="struct"):
        super().__init__()
        self.mod_name = mod_name
        if backbone is None:
            if backbone_name == "gmc":
                backbone = GMC_PL.load_from_checkpoint(ckp)
            elif backbone_name == "clip":
                backbone = CLIP_PL.load_from_checkpoint(ckp)
        self.model = MultiTask_model(
            backbone=backbone, loss_name=loss_name, num_tasks=num_tasks, mod_name=self.mod_name, lr=lr, 
        )
        self.save_hyperparameters()

    def configure_optimizers(self):
        return self.model.optimizer

    def _step(self, batch, batch_idx, step_name, dataloader_idx=None):
        label = batch["labels"]
        probs, logits = self.model._forward_with_sigmoid(
            batch, mod_name=self.mod_name, return_mod="logits"
        )
        loss = self.model.loss(logits, label)

        return loss

    
    def training_step(self, batch, batch_idx):
        loss = self._step(batch=batch, batch_idx=batch_idx, step_name="train")

        batch_size = batch["labels"].shape[0]

        self.log(
            "train/loss",
            loss,
            sync_dist=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )

        return loss
        
    def validation_step(self, batch, batch_idx):
        loss = self._step(
            batch=batch,
            batch_idx=batch_idx,
            step_name="val",
        )

        batch_size = batch["labels"].shape[0]

        self.log(
            "val/loss",
            loss,
            sync_dist=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )

        return loss