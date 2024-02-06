from typing import Dict, Union, List, Tuple, Union
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data

MAX_ATOMIC_NUM = 100

ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}


DEGREES_OFFSET = MAX_ATOMIC_NUM + 1
NUM_DEGREES = 6
CHARGE_OFFSET = DEGREES_OFFSET + NUM_DEGREES + 1
NUM_CHARGE_OPTIONS = 5
CHIRAL_OFFSET = CHARGE_OFFSET + NUM_CHARGE_OPTIONS + 1
NUM_CHIRAL_TAGS = 4
HS_OFFSET = CHIRAL_OFFSET + NUM_CHIRAL_TAGS + 1
NUM_HS_LIMIT = 5;
HYBRIDIZATION_OFFSET = HS_OFFSET + NUM_HS_LIMIT + 1
NUM_HYBRIDIZATION = len(ATOM_FEATURES['hybridization'])
OTHER_OFFSET = HYBRIDIZATION_OFFSET + NUM_HYBRIDIZATION + 1
NUM_OTHER = 2
NUM_TOTAL = OTHER_OFFSET + NUM_OTHER


def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    encoding = [0] * NUM_TOTAL

    value = atom.GetAtomicNum() - 1
    value = value if (value >= 0 and value < MAX_ATOMIC_NUM) else MAX_ATOMIC_NUM
    encoding[value] = 1

    value = atom.GetTotalDegree()
    value = value if (value >= 0 and value < NUM_DEGREES) else NUM_DEGREES
    encoding[value + DEGREES_OFFSET] = 1

    # NOTE: The original had the charges in a different order, but this hopefully shouldn't matter.
    value = atom.GetFormalCharge() + 2
    value = value if (value >= 0 and value < NUM_CHARGE_OPTIONS) else NUM_CHARGE_OPTIONS
    encoding[value + CHARGE_OFFSET] = 1

    value = int(atom.GetChiralTag())
    value = value if (value >= 0 and value < NUM_CHIRAL_TAGS) else NUM_CHIRAL_TAGS
    encoding[value + CHIRAL_OFFSET] = 1

    value = int(atom.GetTotalNumHs())
    value = value if (value >= 0 and value < NUM_HS_LIMIT) else NUM_HS_LIMIT
    encoding[value + HS_OFFSET] = 1

    value = int(atom.GetHybridization())
    choices = ATOM_FEATURES['hybridization']
    value = choices.index(value) if value in choices else NUM_HYBRIDIZATION
    encoding[value + HYBRIDIZATION_OFFSET] = 1

    encoding[OTHER_OFFSET] = 1 if atom.GetIsAromatic() else 0
    encoding[OTHER_OFFSET + 1] = atom.GetMass() * 0.01  # scaled to about the same range as other features

    if functional_groups is not None:
        encoding += functional_groups
    return encoding

# to get rid of using dgllife.utils.one_hot_encoding
def dgl_one_hot_encoding(x, allowable_set):
    return list(map(lambda s: x == s, allowable_set))
    

def mol_to_data(notation, mode="smile", label=None, mol_features=None):
    
    if mode == "smile":
        mol = Chem.MolFromSmiles(notation)
    elif mode == "inchi":
        mol = Chem.inchi.MolFromInchi(notation)
    # else:
    #     print('check notation, mode', notation, mode)
        
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)
    
    src_list = []
    dst_list = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])
    
    node_feature = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()])
    
    edge_feature = []
    type_set = [Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC]
    stereo_set = [Chem.rdchem.BondStereo.STEREONONE,
                  Chem.rdchem.BondStereo.STEREOANY,
                  Chem.rdchem.BondStereo.STEREOZ,
                  Chem.rdchem.BondStereo.STEREOE,
                  Chem.rdchem.BondStereo.STEREOCIS,
                  Chem.rdchem.BondStereo.STEREOTRANS]
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        # feat = one_hot_encoding(bond.GetBondType(), type_set) + [bond.GetIsConjugated(),
        #                                                          bond.IsInRing()] + one_hot_encoding(bond.GetStereo(),
        #                                                                                              stereo_set)
        
        feat = dgl_one_hot_encoding(bond.GetBondType(), type_set) + [bond.GetIsConjugated(),
                                                                     bond.IsInRing()] + dgl_one_hot_encoding(bond.GetStereo(),
                                                                                                             stereo_set)
        
        edge_feature.extend([feat, feat.copy()])
    
    edge_feature = torch.tensor(edge_feature, dtype=torch.float32)
    
    n = Data(x=node_feature, edge_attr=edge_feature, edge_index=torch.tensor([src_list, dst_list], dtype=torch.int64))

    if label is not None:
        n.y = label
    if mol_features is not None:
        n.mol_features = mol_features
    return n


def mol_to_mf(notation,featurizer,mode="smile",output="object"):

    if mode == "smile":
        mol = Chem.MolFromSmiles(notation)
    elif mode == "inchi":
        mol = Chem.MolFromInchi(notation)
    
    mf = featurizer(mol)

    if output == "object":
        return mf
    elif output == "vector":
        return np.array(mf.ToList())
    elif output == "tensor":
        return torch.tensor(np.array(mf.ToList()),dtype=torch.float32)

