import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from data.ssspatch_dataset import SSSPatchDataset
from typing import Optional

#TODO: add transforms to the data?


class SSSPatchDataModule(pl.LightningDataModule):
    def __init__(self,
                 root: str,
                 desc: str = 'sift',
                 img_type: str = 'norm',
                 min_overlap_percentage: float = .15,
                 eval_split: float = .1,
                 batch_size: int = 1,
                 num_workers: int = 1):
        super().__init__()
        self.root = root
        self.desc = desc
        self.img_type = img_type
        self.min_overlap_percentage = min_overlap_percentage
        self.eval_split = eval_split
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        # Set up train and validation datasets
        #TODO: change train to train patches!
        ssspatch_train_full = SSSPatchDataset(self.root, 'test', self.desc,
                                              self.img_type,
                                              self.min_overlap_percentage)
        ssspatch_train_full_len = len(ssspatch_train_full)
        val_len = int(ssspatch_train_full_len * self.eval_split)
        train_len = ssspatch_train_full_len - val_len
        self.ssspatch_train, self.ssspatch_val = random_split(
            ssspatch_train_full, [train_len, val_len],
            generator=torch.Generator().manual_seed(0))

        # Set up test dataset
        self.ssspatch_test = SSSPatchDataset(self.root, 'test', self.desc,
                                             self.img_type,
                                             self.min_overlap_percentage)

    def train_dataloader(self):
        return DataLoader(self.ssspatch_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.ssspatch_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.ssspatch_test, batch_size=self.batch_size, num_workers=self.num_workers)