from typing import Dict, Union, List, Tuple, Union
import pandas as pd
import numpy as np
from random import Random
from math import isclose
from multimodal_contrastive.data.dataset import CustomSubset

import torch.utils.data

from deepchem.splits import (
    RandomSplitter,
    RandomStratifiedSplitter,
    ScaffoldSplitter,
    FingerprintSplitter,
    ButinaSplitter
)
from multimodal_contrastive.data.splitters import ShuffledScaffoldSplitter, SortedScaffoldClusterComboSplitter, SortedClusterSplitter
from rdkit import Chem

MASK = -1
SPLITTERS = {
    "random": RandomSplitter,
    "random_stratified": RandomStratifiedSplitter,
    "scaffold": ScaffoldSplitter,
    "shuffled_scaffold": ShuffledScaffoldSplitter,
    "Sorted_combo": SortedScaffoldClusterComboSplitter,
    "Sorted_cluster": SortedClusterSplitter,
    "butina": ButinaSplitter,
    "fingerprint": FingerprintSplitter,
}

def random_split(dataset, seed, sizes):
    random = Random(seed)

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    train_size = int(sizes[0] * len(dataset))
    train_val_size = int((sizes[0] + sizes[1]) * len(dataset))

    train = CustomSubset(dataset, indices[:train_size])
    val = CustomSubset(dataset, indices[train_size:train_val_size])
    test = CustomSubset(dataset, indices[train_val_size:])
    return train, val, test


def recursive_len(item):
    if type(item) == list:
        return sum(recursive_len(subitem) for subitem in item)
    else:
        return 1


def split_data(dataset, split_type="random", sizes=(0.8, 0.1, 0.1), seed=0, holdout=None, holdout_notion='inchi', holdout_to=None, return_idx=False):
    """
    Split dataset in train/val/test sets, using a specific splitting strategy.
    Each random seed leads to a different split.
    If holdout is given, compoundsin the holdout  and their scaffold will be moved to the test set.

    :param dataset (torch.utils.data.Dataset): Dataset to be split
    :param split_type (string) : Split type, can be random, scaffold, butina, default is random
    :param sizes (tuple): 3-tuple with training/val/test fractions
    :param seed (int): Random seed
    :param holdout (List of inchi string or path of .npy file that hold inchi strings): compounds to be hold out
    :param holdout_notion (string): can be inchi or smiles, default is inchi
    :param holdout_to (string): can be "test", "val", "train" or None, if None, holdout will be dropped
    :return (tuple): 3-tuple with training/val/test datasets
    """
    assert isclose(np.array(sizes).sum(), 1, abs_tol=1e-03)
    print("Train on samples from {}.".format(split_type))
    splitter = SPLITTERS[split_type]()
    
    if holdout is None:
        train_ix, val_ix, test_ix = splitter.split(
            dataset, frac_train=sizes[0], frac_valid=sizes[1], frac_test=sizes[2], seed=seed
        )

    else:
        # remove hold out scaffold from scaffold
        if isinstance(holdout, List):
            smiles_to_holdout = holdout
        elif holdout.endswith('.npy'):
            smiles_to_holdout = np.load(holdout)
        if holdout_notion == 'inchi':
            smiles_to_holdout = [Chem.MolToSmiles(Chem.inchi.MolFromInchi(i)) for i in smiles_to_holdout]
        id_idx_to_holdout = [np.where(np.array(dataset.ids)==_)[0][0] for _ in smiles_to_holdout if _ in set(dataset.ids)]

        if split_type in ["random", "random_stratified"]:
            # remove id_idx_to_holdout from dataset, split, then add back to holdout_to
            tmp_dataset = CustomSubset(dataset, [i for i in range(len(dataset)) if i not in id_idx_to_holdout])
            tmp_dataset.y = tmp_dataset.y.numpy()
            tmp_dataset.w = tmp_dataset.w.numpy()
            train_ix, val_ix, test_ix = splitter.split(
                tmp_dataset, frac_train=sizes[0], frac_valid=sizes[1], frac_test=sizes[2], seed=seed
            )
            if holdout_to is not None:
                assert holdout_to in ['train', 'test', 'val']
                if holdout_to == 'train':
                    train_ix = np.append(train_ix, id_idx_to_holdout)
                elif holdout_to == 'test':
                    test_ix = np.append(test_ix, id_idx_to_holdout)
                else:
                    val_ix = np.append(val_ix, id_idx_to_holdout)
                assert len(train_ix) + len(val_ix) + len(test_ix) == len(tmp_dataset) + len(id_idx_to_holdout)
        else:
            scaffold_sets = splitter.generate_scaffolds(dataset)
            holdout_scaffold_sets = []

            for idx_val in id_idx_to_holdout:
                for scaffold_idx, scaffold_set in enumerate(scaffold_sets):
                    if idx_val in scaffold_set:
                        print('Removing scaffold set: {} ({})'.format(scaffold_idx,scaffold_set))
                        holdout_scaffold_sets.append(scaffold_sets.pop(scaffold_idx))

            # split based on splitter type
            split_mapper = {'train': [], 'test': [], 'val': []}
            if holdout_to is not None:
                assert holdout_to in ['train', 'test', 'val']
                cur_dataset_len = len(dataset)
                print('Adding holdout set to subset {}'.format(holdout_to))
                for h in holdout_scaffold_sets:
                    split_mapper[holdout_to] += h
            else:
                cur_dataset_len = recursive_len(scaffold_sets)
            
            frac_train, frac_valid = sizes[0], sizes[1]
            train_cutoff = frac_train * cur_dataset_len
            valid_cutoff = (frac_train + frac_valid) * cur_dataset_len
                
            train_ix = split_mapper['train']
            val_ix = split_mapper['val']
            test_ix = split_mapper['test']
            for scaffold_set in scaffold_sets:
                if len(train_ix) + len(scaffold_set) > train_cutoff:
                    if len(train_ix) + len(val_ix) + len(scaffold_set) > valid_cutoff:
                        test_ix += scaffold_set
                    else:
                        val_ix += scaffold_set
                else:
                    train_ix += scaffold_set

    train, val, test = (
        CustomSubset(dataset, train_ix),
        CustomSubset(dataset, val_ix),
        CustomSubset(dataset, test_ix),
    )

    if return_idx:
        return train, val, test, train_ix, val_ix, test_ix

    return train, val, test
