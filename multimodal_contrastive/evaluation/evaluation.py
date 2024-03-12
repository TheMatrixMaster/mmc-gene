from typing import Optional, Sequence, Tuple, Union
import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import nn
from torch.optim import Optimizer
from tqdm import trange
import sklearn
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm
from multimodal_contrastive.data.utils import split_data
from torch_geometric.loader import DataLoader
from torch.utils.data.dataloader import default_collate
from sklearn.linear_model import LogisticRegression
import collections
from collections import defaultdict
from sklearn.metrics import roc_auc_score

from collections import defaultdict
from multimodal_contrastive.networks.utils import move_batch_input_to_device
import itertools
from scipy.stats import rankdata
import random

from multimodal_contrastive.analysis.utils import (
    unroll_dataloader,
    get_pairwise_similarity,
    get_values_from_dist_mat,
    make_eval_data_loader
)


def get_rankings(given_emb, target_emb, neg_size=100, metric="cosine", seed=0):
    np.random.seed(seed)
    given_emb = torch.Tensor(given_emb)
    target_emb = torch.Tensor(target_emb)
    batch_size = given_emb.size(0)
    # distance matrix of embeddings of two modality
    if metric == "euclidean":
        dist = torch.cdist(given_emb, target_emb, p=2).cpu().numpy()
    elif metric == "cosine":
        dist = (
            1 - torch.mm(given_emb, target_emb.transpose(0, 1)).detach().cpu().numpy()
        )
    else:
        raise NotImplementedError

    # random sample negative samples at neg_size
    select_idxs = sample_select_idxs(dist, neg_size)

    # get distance matrix of sampled samples (batch_size, neg_size+1)
    sample_dist = np.zeros((dist.shape[0], neg_size + 1))
    for cur_mol_idx in range(dist.shape[0]):
        indices = select_idxs[cur_mol_idx]
        sample_dist[cur_mol_idx] = dist[cur_mol_idx][indices]

    # for target emb that predicted to be top1 similar, its rankings in true ranking

    # use min method, so replicated values will be assigned to the same ranking
    # sorted_indices = np.argsort(sample_dist, axis=1)
    sorted_indices = rankdata(sample_dist, method="min", axis=1)

    all_rankings = []
    for cur_mol_idx in range(dist.shape[0]):
        # use random choice, in case there are multiple replicated distance ranked the 1st
        # updated (need double check)
        all_rankings.append(
            random.choice((sorted_indices[cur_mol_idx] == 1).nonzero()[0]) + 1
        )

    return all_rankings


def process_all_rankings(all_rankings, all_metric_dict, key, r_range=[1, 5, 10]):
    all_rankings = np.array(all_rankings)
    mrr = float(np.mean(1 / all_rankings))
    mrr_std = float(np.std(1 / all_rankings))
    mr = float(np.mean(all_rankings))
    mr_std = float(np.std(all_rankings))
    metric_dict = {"{}_mr".format(key): mr, "{}_mrr".format(key): mrr}
    h_dict = defaultdict(lambda: 0.0)
    for r in r_range:
        h = float(np.mean(all_rankings <= r))
        h_dict["{}_h{}".format(key, r)] = h
    metric_dict.update(h_dict)
    return metric_dict


def sample_select_idxs(sim_mat, neg_size, seed=0):
    np.random.seed(seed)
    select_idxs = []
    for cur_mol_idx in range(sim_mat.shape[0]):
        idxs = np.arange(sim_mat.shape[1]).tolist()
        idxs = idxs[:cur_mol_idx] + idxs[cur_mol_idx + 1 :]
        indices = np.random.choice(idxs, neg_size + 1, replace=False)
        indices[0] = cur_mol_idx
        select_idxs.append(indices)
    return select_idxs


def batch_avg_retrieval_acc(
    model,
    dataset,
    seed,
    retrieval_pairs,
    neg_size=100,
    metric="cosine",
    batch_size=128,
    device="cuda",
    r_range=[1, 5, 10],
):
    test_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=0, pin_memory=False, shuffle=False
    )
    dict_rankings = defaultdict(list)
    all_rankings = []

    for batch in tqdm(test_loader):
        if len(batch["labels"]) < batch_size:
            continue
        x = batch["inputs"]
        x_new = move_batch_input_to_device(x, model.device)

        batch_representations = model.compute_representations(x_new, mod_name=None)

        if retrieval_pairs is None:
            mod_names = list(batch_representations.keys())
            retrieval_pairs = list(itertools.product(mod_names, mod_names))
            for mod in mod_names:
                retrieval_pairs.remove((mod, mod))

        for given, target in retrieval_pairs:
            key = "{}>{}".format(given, target)

            # for infering with multi-mod input
            if "+" in given:
                given_emb = None
                given_name_list = given.split("+")
                for g in given_name_list:
                    if given_emb is None:
                        given_emb = batch_representations[g]
                    else:
                        given_emb += batch_representations[g]
                given_emb /= len(given_name_list)

            else:
                given_emb = batch_representations[given]
            target_emb = batch_representations[target]
            if isinstance(given_emb, collections.abc.Mapping):
                given_emb = torch.mean(torch.stack(list(given_emb.values())), axis=0)
            if isinstance(target_emb, collections.abc.Mapping):
                target_emb = torch.mean(torch.stack(list(target_emb.values())), axis=0)
            rankings = get_rankings(
                given_emb, target_emb, neg_size=neg_size, metric=metric, seed=seed
            )
            dict_rankings[key].extend(rankings)

    all_metric_dict = defaultdict()
    for key, all_rankings in dict_rankings.items():
        metric_dict = process_all_rankings(all_rankings, all_metric_dict, key, r_range)
        all_metric_dict.update(metric_dict)

    return all_metric_dict, retrieval_pairs


def compute_metrics(preds, labels):
    test_ap_list = []
    test_auc_list = []
    for idx in range(0, len(labels)):
        mask = ~np.isnan(labels[idx])
        s1 = set(mask)
        s2 = set(labels[idx][mask])
        if len(s1) == 1 or len(s2) == 1:
            test_ap_list.append(np.nan)
            test_auc_list.append(np.nan)
        else:
            y_true = labels[idx][mask]
            y_scores = preds[idx][mask]
            test_ap = sklearn.metrics.average_precision_score(y_true, y_scores)
            test_auc = sklearn.metrics.roc_auc_score(y_true, y_scores)
            test_ap_list.append(test_ap)
            test_auc_list.append(test_auc)

    return {"auc": np.nanmean(test_auc_list), "ap": np.nanmean(test_ap_list)}


def lp_train(model, data_loader, device, mod_name, max_epochs):
    print("Training linear probing...")
    model = model.to(device)
    model.train()
    optimizer = model.optimizer
    for epoch in range(1, max_epochs):
        for batch in data_loader:
            label = batch["labels"].to(device)
            batch_size = len(label)
            optimizer.zero_grad()
            probs, logits = model._forward_with_sigmoid(
                batch, device=device, return_mod="logits", mod_name=mod_name
            )
            loss = model.loss(logits, label)
            # model.backward(loss)
            loss.backward()
            optimizer.step()


def lp_test(model, data_loader, device, mod_name):
    model.to(device)
    model.eval()
    with torch.no_grad():
        probs, labels = model.predict_probs_dataloader(
            data_loader,
            device=device,
            return_mod="label",
            mod_name=mod_name,
            return_mol=False,
        )
        lp_results = compute_metrics(preds=probs.transpose(), labels=labels.transpose())
    return lp_results


def linear_probe_Kfold(
    model,
    dataset,
    mod_name,
    ckp=None,
    pl_module=None,
    split_type="random",
    init_seed=0,
    n_folds=3,
    max_epochs=200,
):
    all_test_ap = []
    all_test_auc = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in trange(n_folds):
        if ckp is not None:
            model.backbone.load_from_checkpoint(ckp)
        elif pl_module is not None:
            model.backbone.load_state_dict(pl_module.state_dict())
        train_set, val_set, test_set = split_data(
            dataset, split_type=split_type, sizes=(0.8, 0.0, 0.2), seed=init_seed + i
        )
        train_loader = DataLoader(
            train_set, batch_size=128, num_workers=0, pin_memory=False, shuffle=True
        )
        test_loader = DataLoader(
            test_set, batch_size=100, num_workers=0, pin_memory=False, shuffle=False
        )

        lp_train(model, train_loader, device, mod_name, max_epochs)
        lp_results = lp_test(model, test_loader, device, mod_name)
        all_test_ap.append(lp_results["ap"])
        all_test_auc.append(lp_results["auc"])
    return {"auc": np.mean(all_test_auc), "ap": np.mean(all_test_ap)}


class LatentDistCorrelationEvaluator(Callback):
    def __init__(self, model, modes, seed=0):
        self.seed = seed
        self.dataset = None
        self.eval_model = model

        for mode in modes:
            assert mode in ["struct", "ge", "morph", "joint"]

        self.mods = modes
        self.has_struct = "struct" in modes
        self.has_ge = "ge" in modes
        self.has_morph = "morph" in modes
        self.has_joint = "joint" in modes

    def _logCorrelation(self, x, y, x_mode, y_mode, split, pl_module, keep_diag=True) -> None:
        x, y = get_values_from_dist_mat(x, y, keep_diag=keep_diag)
        corr = np.corrcoef(x, y)[0, 1]
        key = "{}~{}~{}".format(x_mode, y_mode, split)
        pl_module.log(key, corr, prog_bar=True, sync_dist=True)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        ok_mods = list(set(self.mods) & set(["ge", "morph"]))

        self.val_dataset = trainer.datamodule.val_dataloader().dataset
        val_loader = make_eval_data_loader(self.val_dataset)
        val_mods = unroll_dataloader(val_loader, mods=ok_mods)

        self.train_dataset = trainer.datamodule.train_dataloader()\
            .dataset.subset(size=len(self.val_dataset))
        train_loader = make_eval_data_loader(self.train_dataset)
        train_mods = unroll_dataloader(train_loader, mods=ok_mods)

        self.test_dataset = trainer.datamodule.test_dataloader().dataset
        test_loader = make_eval_data_loader(self.test_dataset)
        test_mods = unroll_dataloader(test_loader, mods=ok_mods)

        if self.has_ge:
            self.raw_ge_sim_train = get_pairwise_similarity(train_mods['ge'], train_mods['ge'], metric="euclidean", force_positive=True)
            self.raw_ge_sim_val = get_pairwise_similarity(val_mods['ge'], val_mods['ge'], metric="euclidean", force_positive=True)
            self.raw_ge_sim_test = get_pairwise_similarity(test_mods['ge'], test_mods['ge'], metric="euclidean", force_positive=True)

        if self.has_morph:
            pca = PCA(n_components=30)
            # fit pca on train and transform on val and test
            morph_pca_train = pca.fit_transform(train_mods["morph"])
            morph_pca_val = pca.transform(val_mods['morph'])
            morph_pca_test = pca.transform(test_mods['morph'])
            
            self.pca_morph_sim_train = get_pairwise_similarity(morph_pca_train, morph_pca_train, metric="cosine")
            self.pca_morph_sim_val = get_pairwise_similarity(morph_pca_val, morph_pca_val, metric="cosine")
            self.pca_morph_sim_test = get_pairwise_similarity(morph_pca_test, morph_pca_test, metric="cosine")
        
        if self.has_struct:
            # TODO compute tanimoto sim between molecules
            pass

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.eval_model.load_state_dict(pl_module.state_dict())
        self.eval_model.eval()
        self.eval_model.to(device=pl_module.device)

        train_loader = make_eval_data_loader(self.train_dataset)
        val_loader = make_eval_data_loader(self.val_dataset)
        test_loader = make_eval_data_loader(self.test_dataset)
        
        with torch.no_grad():
            train_repr = self.eval_model.compute_representation_dataloader(train_loader, device=pl_module.device)
            val_repr = self.eval_model.compute_representation_dataloader(val_loader, device=pl_module.device)
            test_repr = self.eval_model.compute_representation_dataloader(test_loader, device=pl_module.device)
        
        if self.has_struct and self.has_ge:
            struct_ge_sim_train = get_pairwise_similarity(train_repr["struct"], train_repr["ge"], metric="cosine")
            struct_ge_sim_val = get_pairwise_similarity(val_repr["struct"], val_repr["ge"], metric="cosine")
            struct_ge_sim_test = get_pairwise_similarity(test_repr["struct"], test_repr["ge"], metric="cosine")

            self._logCorrelation(struct_ge_sim_train, self.raw_ge_sim_train, "struct-ge", "raw_ge", "train", pl_module)
            self._logCorrelation(struct_ge_sim_val, self.raw_ge_sim_val, "struct-ge", "raw_ge", "val", pl_module)
            self._logCorrelation(struct_ge_sim_test, self.raw_ge_sim_test, "struct-ge", "raw_ge", "test", pl_module)

        if self.has_joint and self.has_ge:
            joint_ge_sim_train = get_pairwise_similarity(train_repr["joint"], train_repr["ge"], metric="cosine")
            joint_ge_sim_val = get_pairwise_similarity(val_repr["joint"], val_repr["ge"], metric="cosine")
            joint_ge_sim_test = get_pairwise_similarity(test_repr["joint"], test_repr["ge"], metric="cosine")

            self._logCorrelation(joint_ge_sim_train, self.raw_ge_sim_train, "joint-ge", "raw_ge", "train", pl_module)
            self._logCorrelation(joint_ge_sim_val, self.raw_ge_sim_val, "joint-ge", "raw_ge", "val", pl_module)
            self._logCorrelation(joint_ge_sim_test, self.raw_ge_sim_test, "joint-ge", "raw_ge", "test", pl_module)

        if self.has_struct and self.has_morph:
            struct_morph_sim_train = get_pairwise_similarity(train_repr["struct"], train_repr["morph"], metric="cosine")
            struct_morph_sim_val = get_pairwise_similarity(val_repr["struct"], val_repr["morph"], metric="cosine")
            struct_morph_sim_test = get_pairwise_similarity(test_repr["struct"], test_repr["morph"], metric="cosine")

            self._logCorrelation(struct_morph_sim_train, self.pca_morph_sim_train, "struct-morph", "pca_morph", "train", pl_module)
            self._logCorrelation(struct_morph_sim_val, self.pca_morph_sim_val, "struct-morph", "pca_morph", "val", pl_module)
            self._logCorrelation(struct_morph_sim_test, self.pca_morph_sim_test, "struct-morph", "pca_morph", "test", pl_module)

        if self.has_joint and self.has_morph:
            joint_morph_sim_train = get_pairwise_similarity(train_repr["joint"], train_repr["morph"], metric="cosine")
            joint_morph_sim_val = get_pairwise_similarity(val_repr["joint"], val_repr["morph"], metric="cosine")
            joint_morph_sim_test = get_pairwise_similarity(test_repr["joint"], test_repr["morph"], metric="cosine")

            self._logCorrelation(joint_morph_sim_train, self.pca_morph_sim_train, "joint-morph", "pca_morph", "train", pl_module)
            self._logCorrelation(joint_morph_sim_val, self.pca_morph_sim_val, "joint-morph", "pca_morph", "val", pl_module)
            self._logCorrelation(joint_morph_sim_test, self.pca_morph_sim_test, "joint-morph", "pca_morph", "test", pl_module)


class RetrievalOnlineEvaluator(Callback):
    def __init__(self, model, retrieval_pairs=None, neg_size=100, seed=0):
        super().__init__()
        self.seed = seed
        self.retrieval_dataset = None
        self.neg_size = neg_size
        self.eval_model = model
        self.retrieval_pairs = retrieval_pairs

    def set_dataset(self, dataset):
        # randomly sampled K negatives + 1 positive
        self.retrieval_dataset = dataset

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.set_dataset(trainer.datamodule.val_dataloader().dataset)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.eval_model.load_state_dict(pl_module.state_dict())
        self.eval_model.eval()
        self.eval_model.to(device=pl_module.device)

        with torch.no_grad():
            retrieval_results, pairs = batch_avg_retrieval_acc(
                model=self.eval_model,
                dataset=self.retrieval_dataset,
                retrieval_pairs=self.retrieval_pairs,
                seed=self.seed,
                neg_size=self.neg_size,
            )

        for given, target in pairs:
            key = "{}>{}".format(given, target)
            pl_module.log(
                '{}_mrr'.format(key), 
                retrieval_results['{}_mrr'.format(key)], 
                prog_bar=True, 
                sync_dist=True)
            pl_module.log(
                "{}_h1".format(key),
                retrieval_results["{}_h1".format(key)],
                prog_bar=True,
                sync_dist=True,
            )
            pl_module.log(
                "{}_h5".format(key),
                retrieval_results["{}_h5".format(key)],
                prog_bar=True,
                sync_dist=True,
            )
            pl_module.log(
                "{}_h10".format(key),
                retrieval_results["{}_h10".format(key)],
                prog_bar=True,
                sync_dist=True,
            )


class LinearProbeEvaluator(Callback):
    def __init__(self, dataset, model, mod_name, num_folds=10, seed=0, max_epochs=200):
        super().__init__()
        self.seed = seed
        self.num_folds = num_folds
        self.LP_dataset = dataset
        self.LP_model = model
        self.mod_name = mod_name
        self.max_epochs = max_epochs

    def _linear_probe_Kfold(self, ckp=None, state_dict=None, split_type="random"):
        all_test_ap = []
        all_test_auc = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i in trange(self.num_folds):
            if ckp is not None:
                self.LP_model.backbone.load_from_checkpoint(ckp)
            elif state_dict is not None:
                self.LP_model.backbone.load_state_dict(state_dict)
            train_set, val_set, test_set = split_data(
                self.LP_dataset,
                split_type=split_type,
                sizes=(0.8, 0.0, 0.2),
                seed=self.seed + i,
            )
            train_loader = DataLoader(
                train_set, batch_size=128, num_workers=0, pin_memory=False, shuffle=True
            )
            test_loader = DataLoader(
                test_set, batch_size=100, num_workers=0, pin_memory=False, shuffle=False
            )

            lp_train(
                self.LP_model, train_loader, device, self.mod_name, self.max_epochs
            )
            lp_results = lp_test(self.LP_model, test_loader, device, self.mod_name)
            all_test_ap.append(lp_results["ap"])
            all_test_auc.append(lp_results["auc"])
        return {"auc": np.mean(all_test_auc), "ap": np.mean(all_test_ap)}


class LinearProbeOnlineEvaluator(LinearProbeEvaluator):
    def __init__(self, dataset, model, mod_name, num_folds=1, seed=0, max_epochs=50):
        super().__init__(dataset, model, mod_name, num_folds, seed, max_epochs)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        state_dict = pl_module.state_dict()
        lp_results = self._linear_probe_Kfold(
            state_dict=state_dict,
        )
        pl_module.log(
            "LinearProbe_online_auc", lp_results["auc"], prog_bar=True, sync_dist=True
        )
        pl_module.log(
            "LinearProbe_online_ap", lp_results["ap"], prog_bar=True, sync_dist=True
        )


class LinearProbeFinalEvaluator(LinearProbeEvaluator):
    def __init__(self, dataset, model, mod_name, num_folds=1, seed=0, max_epochs=200):
        super().__init__(dataset, model, mod_name, num_folds, seed, max_epochs)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.LP_model.to(device=pl_module.device)
        ckp = trainer.checkpoint_callback.best_model_path
        lp_results = self._linear_probe_Kfold(
            ckp=ckp,
        )
        pl_module.logger.experiment.log({"LinearProbe_auc": lp_results["auc"]})
        pl_module.logger.experiment.log({"LinearProbe_ap": lp_results["ap"]})

