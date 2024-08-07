import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import abc


def get_loss(
    name="gmc",
    *args,
    **kwargs
):
    """Derive the loss function according to loss name.

    Parameters
    ----------
    name (string) : name of the loss
    """
    if name == "gmc":
        return GMCLoss()
    elif name == "clip":
        return CLIPLoss()
    elif name == "mtl_bceloss":
        return MultiTaskLoss()
    elif name == "cross_entropy":
        return MultiClassCrossEntropyLoss()


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()



class GMCLoss(nn.Module):
    """GMCLoss for triple contrastive loss

    Parameters
    ----------
    batch_representations : batch list representations of each modality
    temp: temperature parameter for calculating the loss
    batch_size: batch size
    """

    def __init__(self):
        super().__init__()

    def forward(self, batch_representations, temp, batch_size, learnable_w=None):
        loss_list = gmc_loss_terms(batch_representations, temp, batch_size)
        if learnable_w is None:
            loss_total = torch.mean(torch.sum(torch.stack(loss_list), dim=0), dim=0)
        else:
            loss_total = torch.mean(
                torch.sum((torch.vstack(loss_list).T * learnable_w).T, dim=0), dim=0
            )
        return loss_total


class CLIPLoss(nn.Module):
    """CLIPLoss for dual contrastive loss

    Parameters
    ----------
    batch_representations : batch list representations of each modality
    temp: temperature parameter for calculating the loss
    batch_size: batch size
    normalize: if normalize the input representations before calculating contrastive loss
    """

    def __init__(self):
        super().__init__()

    def forward(self, batch_representations, temp, batch_size, normalize=True):
        emb_x, emb_y = list(batch_representations.values())
        CL_loss_1 = do_CL(emb_x, emb_y, normalize, temp, batch_size)
        CL_loss_2 = do_CL(emb_y, emb_x, normalize, temp, batch_size)
        loss = (CL_loss_1 + CL_loss_2) / 2
        return loss


def do_CL(X, Y, normalize, temp, batch_size):
    if normalize:
        X = F.normalize(X)
        Y = F.normalize(Y)

    criterion = nn.CrossEntropyLoss()
    B = batch_size

    logits = torch.mm(X, Y.transpose(1, 0))  # B*B

    logits = torch.div(logits, temp)
    labels = torch.arange(B).long().to(logits.device)  # B*1

    CL_loss = criterion(logits, labels)

    return CL_loss


def do_gmc_loss(batch_representations, temperature, batch_size):
    batch_representations = list(batch_representations.values())

    joint_mod_loss_sum = 0
    for mod in range(len(batch_representations) - 1):
        # Negative pairs: everything that is not in the current joint-modality pair
        out_joint_mod = torch.cat(
            [batch_representations[-1], batch_representations[mod]], dim=0
        )
        # [2*B, 2*B]
        sim_matrix_joint_mod = torch.exp(
            torch.mm(out_joint_mod, out_joint_mod.t().contiguous()) / temperature
        )
        # Mask for remove diagonal that give trivial similarity, [2*B, 2*B]
        mask_joint_mod = (
            torch.ones_like(sim_matrix_joint_mod)
            - torch.eye(2 * batch_size, device=sim_matrix_joint_mod.device)
        ).bool()
        # Remove 2*B diagonals and reshape to [2*B, 2*B-1]
        sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(mask_joint_mod).view(
            2 * batch_size, -1
        )

        # Positive pairs: cosine loss joint-modality
        pos_sim_joint_mod = torch.exp(
            torch.sum(batch_representations[-1] * batch_representations[mod], dim=-1)
            / temperature
        )
        # [2*B]
        pos_sim_joint_mod = torch.cat([pos_sim_joint_mod, pos_sim_joint_mod], dim=0)
        loss_joint_mod = -torch.log(
            pos_sim_joint_mod / sim_matrix_joint_mod.sum(dim=-1)
        )
        joint_mod_loss_sum += loss_joint_mod

    loss = torch.mean(joint_mod_loss_sum)

    return loss


def gmc_loss_terms(batch_representations, temperature, batch_size):
    # if mod_representation is a dictionary of reprs of different replicates
    for mod, reprs in batch_representations.items():
        if isinstance(reprs, abc.Mapping) == True:
            batch_representations.update(
                {mod: torch.mean(torch.stack(list(reprs.values())), axis=0)}
            )

    batch_representations = list(batch_representations.values())

    joint_mod_loss_terms = []
    for mod in range(len(batch_representations) - 1):
        # Negative pairs: everything that is not in the current joint-modality pair
        out_joint_mod = torch.cat(
            [batch_representations[-1], batch_representations[mod]], dim=0
        )
        # [2*B, 2*B]
        sim_matrix_joint_mod = torch.exp(
            torch.mm(out_joint_mod, out_joint_mod.t().contiguous()) / temperature
        )
        # Mask for remove diagonal that give trivial similarity, [2*B, 2*B]
        mask_joint_mod = (
            torch.ones_like(sim_matrix_joint_mod)
            - torch.eye(2 * batch_size, device=sim_matrix_joint_mod.device)
        ).bool()
        # Remove 2*B diagonals and reshape to [2*B, 2*B-1]
        sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(mask_joint_mod).view(
            2 * batch_size, -1
        )

        # Positive pairs: cosine loss joint-modality
        pos_sim_joint_mod = torch.exp(
            torch.sum(batch_representations[-1] * batch_representations[mod], dim=-1)
            / temperature
        )
        # [2*B]
        pos_sim_joint_mod = torch.cat([pos_sim_joint_mod, pos_sim_joint_mod], dim=0)
        loss_joint_mod = -torch.log(
            pos_sim_joint_mod / sim_matrix_joint_mod.sum(dim=-1)
        )
        joint_mod_loss_terms.append(loss_joint_mod)

    return joint_mod_loss_terms


class MultiTaskLoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, output, label):
        # Create a mask to identify where labels are not NaN
        mask = ~torch.isnan(label)

        # Select only the elements corresponding to non-NaN labels
        valid_output = torch.masked_select(output, mask)
        valid_label = torch.masked_select(label, mask)

        # Use BCELoss to compute the loss on the valid elements
        valid_label = valid_label.to(torch.float32)
        valid_output = valid_output.to(torch.float32)

        loss = F.binary_cross_entropy_with_logits(
            valid_output, valid_label, reduction=self.reduction
        )

        return loss, valid_output, valid_label


class MultiClassCrossEntropyLoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, output, label):
        valid_label = label.to(torch.long)
        valid_output = output.to(torch.float32)
        loss = F.cross_entropy(valid_output, valid_label, reduction=self.reduction)
        return loss, valid_output, valid_label
