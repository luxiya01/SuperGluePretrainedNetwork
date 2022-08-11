from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

from data.ssspatch_dataset import SSSPatchDataset


# TODO: add transforms to the data?


class SSSPatchDataModule(pl.LightningDataModule):
    def __init__(self,
                 config):
        super().__init__()
        self.root = config.data_root
        # TODO: allow for selection of descriptors!
        self.num_kps = config.data_num_kps
        self.img_type = config.data_img_type
        self.min_overlap = config.data_min_overlap
        self.eval_split = config.data_eval_split
        self.batch_size = config.data_batch_size
        self.num_workers = config.data_num_workers
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('SSSPatchDataModule')
        parser.add_argument('--data_root', type=str, default='')
        parser.add_argument('--data_num_kps', type=int, default=100)
        parser.add_argument('--data_img_type', type=str, default='norm')
        parser.add_argument('--data_min_overlap', type=float, default=.15)
        parser.add_argument('--data_eval_split', type=float, default=.1)
        parser.add_argument('--data_batch_size', type=int, default=1)
        parser.add_argument('--data_num_workers', type=int, default=1)
        return parent_parser

    def setup(self, stage: Optional[str] = None) -> None:
        # Set up train and validation datasets
        # TODO: change train to train patches!
        ssspatch_train_full = SSSPatchDataset(root=self.root, img_type=self.img_type, num_kps=self.num_kps,
                                              min_overlap_percentage=self.min_overlap, train=False)
        ssspatch_train_full_len = len(ssspatch_train_full)
        val_len = int(ssspatch_train_full_len * self.eval_split)
        train_len = ssspatch_train_full_len - val_len
        self.ssspatch_train, self.ssspatch_val = random_split(
            ssspatch_train_full, [train_len, val_len],
            generator=torch.Generator().manual_seed(0))

        # Set up test dataset
        self.ssspatch_test = SSSPatchDataset(root=self.root,
                                             img_type=self.img_type, num_kps=self.num_kps,
                                             min_overlap_percentage=self.min_overlap, train=False)

    def train_dataloader(self):
        return DataLoader(self.ssspatch_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.ssspatch_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.ssspatch_test, batch_size=self.batch_size, num_workers=self.num_workers)
