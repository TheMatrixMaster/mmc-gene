import torch
import numpy as np
from collections import defaultdict
from torch_geometric.loader import DataLoader
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from multimodal_contrastive.networks.utils import move_batch_input_to_device

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs


def random_subset(mols, representations, cutoff=3000):
    idx = []
    if len(mols) > cutoff:
        idx = np.random.choice(len(mols), cutoff, replace=False)
        pMols = np.array(mols)[idx]
        pRepresentations = {k: np.array(v)[idx] for k, v in representations.items()}
    else:
        idx = np.arange(len(mols))
        pMols = np.array(mols)
        pRepresentations = representations

    return pMols, pRepresentations, idx


def make_eval_data_loader(dataset, batch_size=128):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
    )


def unroll_dataloader(dataloader, mods=["ge"], device="cpu"):
    outs = defaultdict(list)
    with torch.no_grad():
        for batch in tqdm(dataloader, position=0, leave=True):
            x_dict = batch["inputs"]
            # x_dict = move_batch_input_to_device(x_dict, device=device)
            for mod in mods:
                assert mod in x_dict.keys()
                outs[mod].append(x_dict[mod])

    final_outs = dict()
    for mod in outs.keys():
        final_outs[mod] = np.vstack(outs[mod])

    return final_outs


def save_representations(representations, mols, path):
    assert representations["struct"].shape[0] == len(mols)
    np.savez(
        path,
        struct=representations["struct"],
        morph=representations["morph"] if "morph" in representations else None,
        ge=representations["ge"] if "ge" in representations else None,
        mols=mols,
    )


def get_pairwise_similarity(modx, mody, metric="euclidean", force_positive=False):
    assert modx.shape[1] == mody.shape[1]
    assert len(modx) == len(mody)

    if metric == "euclidean":
        if force_positive:
            return 1 / np.exp(euclidean_distances(modx, mody))
        else:
            return euclidean_distances(modx, mody)
    elif metric == "cosine":
        return cosine_similarity(modx, mody)
    elif metric == "mse":
        mse = np.zeros((idx, idx), dtype=float)
        for i1 in tqdm(range(idx)):
            for i2 in range(idx):
                mse[i1, i2] = np.mean((ge[i1] - ge[i2]) ** 2)
        return mse
    else:
        raise ValueError(f"Unknown metric: {metric}")


def get_molecular_fingerprints(mols, fp_type="morgan", radius=2, nbits=2048):
    fps = []
    if fp_type == "morgan":
        fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=nbits)
        for mol in tqdm(mols):
            mol = Chem.MolFromSmiles(mol)
            fp = fpgen.GetSparseCountFingerprint(mol)
            fps.append(fp)
    else:
        raise ValueError(f"Unknown fp_type: {fp_type}")

    return fps


def get_molecular_similarity(fps, metric="tanimoto"):
    # Compute molecular similarity between all fingerprints
    num_mols = len(fps)
    sim = np.zeros((num_mols, num_mols))

    for i in tqdm(range(num_mols)):
        for j in range(i, num_mols):
            if metric == "tanimoto":
                sim[i, j] = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            elif metric == "dice":
                sim[i, j] = DataStructs.DiceSimilarity(fps[i], fps[j])
            else:
                raise ValueError(f"Unknown metric: {metric}")

            sim[j, i] = sim[i, j]

    return sim


def get_values_from_dist_mat(x, y, keep_diag=False):
    # check if x, y are symmetric matrices
    assert x.shape == y.shape
    is_x_symmetric = np.allclose(x, x.T)
    is_y_symmetric = np.allclose(y, y.T)

    if not is_x_symmetric or not is_y_symmetric:
        # if either dist matrix is asymmetric, we need to keep all values and just flatten matrices
        return np.ravel(x), np.ravel(y)

    if keep_diag:
        # keep upper triangle including diagonal
        idx = np.triu_indices(x.shape[0])
    else:
        # keep upper triangle excluding diagonal
        idx = np.triu_indices(x.shape[0], k=1)

    return x[idx], y[idx]
