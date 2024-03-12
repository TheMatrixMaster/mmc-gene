from typing import List, Tuple, Optional
import wandb
import numpy as np
from deepchem.data import Dataset
from deepchem.splits.splitters import ScaffoldSplitter, _generate_scaffold, logger

class ShuffledScaffoldSplitter(ScaffoldSplitter):
  """Class for doing data splits based on the scaffold of small molecules.
  
  Same as the ScaffoldSplitter class from deepchem.splits.splitters, except
  that this version does not sort scaffolds in decreasing order. Instead it uses
  a random seed to shuffle the scaffold set order.
  """

  def split(
      self,
      dataset: Dataset,
      frac_train: float = 0.8,
      frac_valid: float = 0.1,
      frac_test: float = 0.1,
      seed: Optional[int] = None,
  ) -> Tuple[List[int], List[int], List[int]]:
    """
    Splits internal compounds into train/validation/test by scaffold.

    Parameters
    ----------
    dataset: Dataset
      Dataset to be split.
    frac_train: float, optional (default 0.8)
      The fraction of data to be used for the training split.
    frac_valid: float, optional (default 0.1)
      The fraction of data to be used for the validation split.
    frac_test: float, optional (default 0.1)
      The fraction of data to be used for the test split.
    seed: int, optional (default None)
      Random seed to use.

    Returns
    -------
    Tuple[List[int], List[int], List[int]]
      A tuple of train indices, valid indices, and test indices.
      Each indices is a list of integers.
    """
    if seed is not None:
      np.random.seed(seed)

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    scaffold_sets = self.generate_scaffolds(dataset)

    train_cutoff = frac_train * len(dataset)
    valid_cutoff = (frac_train + frac_valid) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    num_scaffolds_train, num_scaffolds_val, num_scaffolds_test = 0, 0, 0

    logger.info("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
      if len(train_inds) + len(scaffold_set) > train_cutoff:
        if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
          test_inds += scaffold_set
          num_scaffolds_test += 1
        else:
          valid_inds += scaffold_set
          num_scaffolds_val += 1
      else:
        train_inds += scaffold_set
        num_scaffolds_train += 1

    logger.info("Assigned %d scaffold sets to train" % num_scaffolds_train)
    logger.info("Assigned %d scaffold sets to valid" % num_scaffolds_val)
    logger.info("Assigned %d scaffold sets to test" % num_scaffolds_test)

    wandb.log({"num_scaffolds_train": num_scaffolds_train, 
               "num_scaffolds_val": num_scaffolds_val,
               "num_scaffolds_test": num_scaffolds_test})

    return train_inds, valid_inds, test_inds

  def generate_scaffolds(self,
                         dataset: Dataset,
                         log_every_n: int = 1000) -> List[List[int]]:
    """Returns all scaffolds from the dataset.

    Parameters
    ----------
    dataset: Dataset
      Dataset to be split.
    log_every_n: int, optional (default 1000)
      Controls the logger by dictating how often logger outputs
      will be produced.

    Returns
    -------
    scaffold_sets: List[List[int]]
      List of indices of each scaffold in the dataset.
    """
    scaffolds = {}
    data_len = len(dataset)

    logger.info("About to generate scaffolds")
    for ind, smiles in enumerate(dataset.ids):
      if ind % log_every_n == 0:
        logger.info("Generating scaffold %d/%d" % (ind, data_len))
      scaffold = _generate_scaffold(smiles)
      if scaffold not in scaffolds:
        scaffolds[scaffold] = [ind]
      else:
        scaffolds[scaffold].append(ind)

    # Shuffle the scaffold set order randomly (not based on set size)
    scaffolds = { key: sorted(value) for key, value in scaffolds.items() }
    scaffold_sets = list(scaffolds.values())
    np.random.shuffle(scaffold_sets)
    return scaffold_sets
  