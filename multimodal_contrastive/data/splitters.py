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

        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
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

        if wandb.run is not None:
            wandb.log(
                {
                    "num_scaffolds_train": num_scaffolds_train,
                    "num_scaffolds_val": num_scaffolds_val,
                    "num_scaffolds_test": num_scaffolds_test,
                }
            )

        return train_inds, valid_inds, test_inds

    def generate_scaffolds(
        self, dataset: Dataset, log_every_n: int = 1000
    ) -> List[List[int]]:
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
        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        scaffold_sets = list(scaffolds.values())
        np.random.shuffle(scaffold_sets)
        return scaffold_sets


def _reorder_index(index_sets, test_size, val_size, balance=True, seed=0):
    if balance:
        print('Using balanced reorder...')
        # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        np.random.seed(seed)
        np.random.shuffle(big_index_sets)
        np.random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(
            index_sets,
            key=lambda index_set: len(index_set),
            reverse=True,
        )

    return index_sets


def _reorder_index(index_sets, test_size, val_size, balance=True, seed=0):
    if balance:
        print('Using balanced reorder...')
        # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        np.random.seed(seed)
        np.random.shuffle(big_index_sets)
        np.random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(
            index_sets,
            key=lambda index_set: len(index_set),
            reverse=True,
        )

    return index_sets


def _split_sets(index_sets, train_cutoff, valid_cutoff):

    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    num_train, num_val, num_test = 0, 0, 0

    logger.info("About to sort in index sets")
    for index_set in index_sets:
        if len(train_inds) + len(index_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(index_set) > valid_cutoff:
                test_inds += index_set
                num_test += 1
            else:
                valid_inds += index_set
                num_val += 1
        else:
            train_inds += index_set
            num_train += 1
            
    return train_inds, valid_inds, test_inds


class SortedScaffoldClusterComboSplitter(ScaffoldSplitter):
    """Class for doing data splits based on the scaffold and cluster of small molecules.
    """
    def generate_scaffolds(
        self, dataset: Dataset, log_every_n: int = 1000
    ) -> List[List[int]]:
        """Returns all scaffolds+cluster combos from the dataset.

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
          List of indices of each combo in the dataset.
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
        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        scaffold_sets = list(scaffolds.values())
        np.random.shuffle(scaffold_sets)
        return scaffold_sets


    def generate_clusters(
        self, dataset: Dataset, log_every_n: int = 1000
    ) -> List[List[int]]:
        """Returns all clusters from the dataset.

        Parameters
        ----------
        dataset: Dataset
          Dataset to be split.
        log_every_n: int, optional (default 1000)
          Controls the logger by dictating how often logger outputs
          will be produced.

        Returns
        -------
        cluster_sets: List[List[int]]
          List of indices of each scaffold in the dataset.
        """
        clusters = {}
        data_len = len(dataset)

        logger.info("About to generate clusters")
        for ind, smiles in enumerate(dataset.ids):
            if ind % log_every_n == 0:
                logger.info("Generating scaffold %d/%d" % (ind, data_len))
            cluster = dataset.clusters[ind]
            if cluster not in clusters:
                clusters[cluster] = [ind]
            else:
                clusters[cluster].append(ind)

        # # Shuffle the scaffold set order randomly (not based on set size)
        # cluster = {key: sorted(value) for key, value in clusters.items()}
        cluster_sets = list(clusters.values())
        # np.random.shuffle(cluster_sets)
        return cluster_sets

    def split(
        self,
        dataset: Dataset,
        frac_train: float = 0.8,
        frac_valid: float = 0.1,
        frac_test: float = 0.1,
        seed: Optional[int] = None,
        balance=False,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Splits internal compounds into train/validation/test by scaffold+cluster.

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

        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        scaffold_sets = self.generate_scaffolds(dataset)
        cluster_sets = self.generate_clusters(dataset)

        train_cutoff = frac_train * len(dataset)
        valid_cutoff = (frac_train + frac_valid) * len(dataset)

        val_size, test_size = (
        int(frac_valid * len(dataset)),
        int(frac_test * len(dataset)),
    )

        scaffold_sets = _reorder_index(scaffold_sets, seed=seed, balance=balance, test_size=test_size, val_size=val_size)
        train_inds_sc, valid_inds_sc, test_inds_sc = _split_sets(index_sets=scaffold_sets, train_cutoff=train_cutoff, valid_cutoff=valid_cutoff)

        cluster_sets = _reorder_index(cluster_sets, seed=seed, balance=balance, test_size=test_size, val_size=val_size)
        train_inds_cl, valid_inds_cl, test_inds_cl = _split_sets(index_sets=cluster_sets, train_cutoff=train_cutoff, valid_cutoff=valid_cutoff)

        train_inds = list(set(train_inds_sc) & set(train_inds_cl))
        valid_inds = list(set(valid_inds_sc) & set(valid_inds_cl))
        test_inds = list(set(test_inds_sc) & set(test_inds_cl))
        total = len(train_inds)+len(valid_inds)+len(test_inds)
        
        logger.info('Train: {} {}%'.format(len(train_inds), (len(train_inds)/total)))
        logger.info('Valid: {} {}%'.format(len(valid_inds), (len(valid_inds)/total)))
        logger.info('Test: {} {}%'.format(len(test_inds), (len(test_inds)/total)))
        logger.info('Total: {} {}%'.format(total, total/len(dataset)))

        return train_inds, valid_inds, test_inds


class SortedClusterSplitter(ScaffoldSplitter):
    """Class for doing data splits based on the cluster of small molecules.
    """

    def generate_clusters(
        self, dataset: Dataset, log_every_n: int = 1000
    ) -> List[List[int]]:
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
          List of indices of each cluster in the dataset.
        """
        clusters = {}
        data_len = len(dataset)

        logger.info("About to generate scaffolds")
        for ind, smiles in enumerate(dataset.ids):
            if ind % log_every_n == 0:
                logger.info("Generating scaffold %d/%d" % (ind, data_len))
            cluster = dataset.clusters[ind]
            if cluster not in clusters:
                clusters[cluster] = [ind]
            else:
                clusters[cluster].append(ind)

        # Shuffle the scaffold set order randomly (not based on set size)
        # cluster = {key: sorted(value) for key, value in clusters.items()}
        cluster_sets = list(clusters.values())
        # np.random.shuffle(cluster_sets)
        return cluster_sets

    def split(
        self,
        dataset: Dataset,
        frac_train: float = 0.8,
        frac_valid: float = 0.1,
        frac_test: float = 0.1,
        seed: Optional[int] = None,
        balance: str = False,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Splits internal compounds into train/validation/test by cluster.

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

        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        cluster_sets = self.generate_clusters(dataset)

        train_cutoff = frac_train * len(dataset)
        valid_cutoff = (frac_train + frac_valid) * len(dataset)

        val_size, test_size = (
        int(frac_valid * len(dataset)),
        int(frac_test * len(dataset)),
    )

        cluster_sets = _reorder_index(cluster_sets, seed=seed, balance=balance, test_size=test_size, val_size=val_size)
        train_inds, valid_inds, test_inds = _split_sets(index_sets=cluster_sets, train_cutoff=train_cutoff, valid_cutoff=valid_cutoff)
    
        total = len(train_inds)+len(valid_inds)+len(test_inds)
        logger.info('Train: {} {}%'.format(len(train_inds), (len(train_inds)/total)))
        logger.info('Valid: {} {}%'.format(len(valid_inds), (len(valid_inds)/total)))
        logger.info('Test: {} {}%'.format(len(test_inds), (len(test_inds)/total)))   
        logger.info('Total: {} {}%'.format(total, total/len(dataset)))

        return train_inds, valid_inds, test_inds