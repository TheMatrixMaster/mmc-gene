from typing import Dict, Union, List, Tuple, Union
import torch
from torch.utils.data import Dataset, Subset
import pandas as pd
import numpy as np
from ..data.featurization import mol_to_data, mol_to_mf
from collections import OrderedDict
import h5py
from random import Random
from rdkit import Chem
from sklearn import preprocessing
import pickle
import re
from rdkit.Chem import rdFingerprintGenerator

class H5Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        mods,
        labels=None,
        joint_as_input=False,
        cluster: str=None,
    ):
        self.f = h5py.File(data, "r")
        if "valid_smiles" in self.f.keys():
            self.ids = self.f["valid_smiles"][...]
        else:
            self.mols = self.f["valid_mols"][...]
            self.ids = [Chem.MolToSmiles(Chem.inchi.MolFromInchi(i)) for i in self.mols]
        self.mods = mods
        self.joint_as_input = joint_as_input

        self.y = labels
        if self.y is None:
            self.y = torch.Tensor([-1])

        if cluster is not None:
            with open(cluster, 'rb') as handle:
                self.clusters = pickle.load(handle)

    def subset(self, size=3000):
        idx = np.random.choice(len(self.ids), size, replace=False)
        return CustomSubset(self, idx)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # ipt_mods is the unmasked original
        ipt_mods = OrderedDict()
        for mod in self.mods:
            feat = self._get_mod_feat(idx, mod)
            ipt_mods.update({mod: feat})

        if self.joint_as_input:
            joint = OrderedDict()
            ipt_mods.update(
                {"joint": OrderedDict([(mod, ipt_mods[mod]) for mod in self.mods])}
            )

        return OrderedDict(
            [
                ("inputs", ipt_mods),
                ("labels", self.y if len(self.y) == 1 else self.y[idx]),
            ]
        )

    def _get_mod_feat(self, idx, mod):
        raise NotImplementedError


class H5DatasetJUMP(H5Dataset):
    def __init__(
        self,
        data,
        mods=["struct", "morph"],
        struct_ipt="graph",
        labels=None,
        joint_as_input=True,
        morph_mode="mean",
        avg_morph_feat=True,
        cluster: str=None,
    ):
        super().__init__(
            data,
            mods,
            labels,
            joint_as_input,
            cluster,
        )
        self.morph_mode = morph_mode
        self.struct_ipt = struct_ipt
        if self.struct_ipt == "mf":
            self.mf_featurizer = dc.feat.CircularFingerprint(size=1024, radius=3)

        self.data_morph = self.f["morph"]
        self.morph_idx = self.f["morph_index"]
        self.morph_meta = self.f["morph_metadata"]
        self.avg_morph_feat = avg_morph_feat

    def _get_mod_feat(self, idx, mod):
        mol = self.ids[idx]
        if mod == "struct":
            if self.struct_ipt == "graph":
                feat = mol_to_data(mol, mode="smile", label=None, mol_features=None)
                feat.mols = mol
            elif self.struct_ipt == "mf":
                feat = mol_to_mf(mol, self.mf_featurizer, mode="smile")
                
        elif mod == "morph":
            mol_sidx, mol_eidx = self.morph_idx[idx]
            morph_feat_pool = self.data_morph[mol_sidx:mol_eidx, :]
            num_replicates = morph_feat_pool.shape[0]

            if self.avg_morph_feat:
                feat = torch.FloatTensor(morph_feat_pool.mean(0))

            if self.morph_mode == "random":
                ix = np.random.choice(num_replicates, size=1, replace=False)
                feat = torch.FloatTensor(morph_feat_pool[ix, :].flatten())

            elif self.morph_mode == "mean":
                feat = torch.FloatTensor(morph_feat_pool.mean(0))
                
        else:
            raise ValueError("Modality {} not implemented".format(mod))

        return feat


class H5DatasetPUMA(H5Dataset):
    """This is a dataset class for PUMA dataset.

    Parameters:
    data (h5 file path):
    labels: list of labels for supervised task/loss, default is None
    """

    def __init__(self, data, labels=None, mods=None, joint_as_input=False, cluster: str=None,):
        super().__init__(
            data,
            mods,
            labels,
            joint_as_input,
            cluster, 
        )
        self.data_morph = preprocessing.normalize(self.f["morph"][...], axis=1)
        self.data_ge = preprocessing.normalize(self.f["ge"][...], axis=1)

    def _get_mod_feat(self, idx, mod):
        mol = self.ids[idx]
        if mod == "struct":
            feat = mol_to_data(
                notation=mol,
                mode="smile",
                label=None,
                mol_features=None,
            )
            feat.mols = mol

        elif mod == "morph":
            feat = torch.FloatTensor(self.data_morph[idx, :])
        elif mod == "ge":
            feat = torch.FloatTensor(self.data_ge[idx, :])
        else:
            raise ValueError("Modality {} not implemented".format(mod))

        return feat


class TestDataset(torch.utils.data.Dataset):
    """This is a dataset class for test dataset for linear evaluation.

    Parameters:
    path (csv): csv file with smiles and assay labels
    """

    def __init__(
        self,
        data,
        type='fp',
        mol_col="valid_smiles",
        label_col=None,
        sample_ratio=None,
        hold_out_indices=None,
        device=None,
        seed=0,
    ):
        df = pd.read_csv(data)
        mols = df[mol_col].values
        if label_col is not None:
            labels = df[label_col].values
        else:
            label_names = [x for x in df.columns if re.match('^\d', x)]
            labels = df[label_names].to_numpy()
        if sample_ratio is not None:
            subset_size = int(df.shape[0] * sample_ratio)
            indices = list(range(df.shape[0]))
            random = Random(seed)
            random.shuffle(indices)
            filter = indices[:subset_size]

            mols = mols[filter]
            labels = labels[filter]

        self.ids = mols
        self.y = torch.Tensor(labels)
        self.w = 1 - torch.isnan(self.y).int()
        
        if self.y is None:
            self.y = torch.Tensor([-1])
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.type = type
        if self.type == 'fp':
            fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
            self.mf_featurizer = fpgen.GetCountFingerprint

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        smile = self.ids[idx]

        if self.type == "graph":
            struct_feat = self.smile_to_data(
                smile=smile,
                label=torch.Tensor(self.y[idx]).float() if self.y is not None else None,
                mol_features=None,
            )
            struct_feat.mols = smile
        elif self.type == 'fp':
            struct_feat = mol_to_mf(smile, self.mf_featurizer, mode='smile', output="tensor")

        return {
            "inputs": {
                "struct": struct_feat,
            },
            "labels": self.y[idx, :],
        }

    @staticmethod
    def smile_to_data(smile, mode="smile", label=None, mol_features=None):
        return mol_to_data(smile, mode=mode, label=label, mol_features=mol_features)


class CustomSubset(Subset):
    """This is a Subset class for splited dataset

    Parameters:
    dataset:
    indicies:
    """

    def __init__(self, dataset, indices, filter_source=None):
        super().__init__(dataset, indices)
        self.ids = (
            self.dataset.ids[self.indices] if self.dataset.ids is not None else None
        )
        if self.dataset.y.shape[0] == self.dataset.ids.shape[0]:
            self.y = self.dataset.y[self.indices]
            self.w = self.dataset.w[self.indices]
        else:
            self.y = None
            self.w = None
        self.filter_source = filter_source

    def __getitem__(self, idx):
        self.dataset.filter_source = self.filter_source
        item = super().__getitem__(idx)
        return item

    def __len__(self):
        return len(self.ids)

    def __getattr__(self, item):
        return getattr(self.dataset, item)
