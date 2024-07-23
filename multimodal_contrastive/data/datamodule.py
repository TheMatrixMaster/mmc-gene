from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from ..data.utils import split_data


class MultiInput_DataModule(pl.LightningDataModule):
    """This is a datamodule class

    Parameters: 
    split_sizes: 3-tuple of float that defines the size of train, val, test dataset
    """

    def __init__(
        self,
        dataset,
        batch_size=128,
        num_workers=8,
        pin_memory=False,
        split_sizes=(0.8, 0.1, 0.1),
        split_type="random",
        holdout=None,
        holdout_notion="inchi",
        holdout_to=None,
        seed=0,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.split_sizes = split_sizes
        self.split_type = split_type
        self.seed = seed
        self.holdout = holdout
        self.holdout_notion = holdout_notion
        self.holdout_to = holdout_to

    def setup(self, stage: str):
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            self.train_idx,
            self.val_idx,
            self.test_idx,
        ) = split_data(
            self.dataset,
            split_type=self.split_type,
            sizes=self.split_sizes,
            seed=self.seed,
            holdout=self.holdout,
            holdout_notion=self.holdout_notion,
            holdout_to=self.holdout_to,
            return_idx=True,
        )

        print("Train on {} samples.".format(len(self.train_dataset)))
        print("Validate on {} samples.".format(len(self.val_dataset)))
        print("Test on {} samples.".format(len(self.test_dataset)))

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=False,
            drop_last=True,
            # collate_fn=self.Collator,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=False,
            drop_last=True,
        )

    def infer_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
        )

    def get_split_idx(self):
        return self.train_idx, self.val_idx, self.test_idx
