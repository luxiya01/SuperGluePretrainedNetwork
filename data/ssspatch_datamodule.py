from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms

from data.image_transforms import ColumnwiseNormalization
from data.ssspatch_dataset import SSSPatchDataset


# TODO: add transforms to the data?


class SSSPatchDataModule(pl.LightningDataModule):
    def __init__(self,
                 root: str, num_kps: int = 100, min_overlap: float = .15, eval_split: float = .1,
                 batch_size: int = 1, num_workers: int = 1, train_image_transform: list = ['column_norm'],
                 test_image_transform: list = ['column_norm'], data_kwargs: dict = None):
        super().__init__()
        self.root = root
        self.num_kps = num_kps
        self.min_overlap = min_overlap
        self.eval_split = eval_split
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_image_transform = self._setup_transforms(train_image_transform, data_kwargs)
        self.test_image_transform = self._setup_transforms(test_image_transform, data_kwargs)

        self.save_hyperparameters()

    @staticmethod
    def _setup_transforms(transform_names: list, kwargs: dict):
        tf = []
        for name in transform_names:
            if name == 'column_norm':
                tf.append(ColumnwiseNormalization(a_max=kwargs['a_max']))
        return transforms.Compose(tf)

    def setup(self, stage: Optional[str] = None) -> None:
        # Set up train and validation datasets
        ssspatch_train_full = SSSPatchDataset(root=self.root, num_kps=self.num_kps,
                                              min_overlap_percentage=self.min_overlap, train=True,
                                              transform=self.train_image_transform)
        ssspatch_train_full_len = len(ssspatch_train_full)
        val_len = int(ssspatch_train_full_len * self.eval_split)
        train_len = ssspatch_train_full_len - val_len
        self.ssspatch_train, self.ssspatch_val = random_split(
            ssspatch_train_full, [train_len, val_len],
            generator=torch.Generator().manual_seed(0))

        # Set up test dataset
        self.ssspatch_test = SSSPatchDataset(root=self.root,
                                             num_kps=self.num_kps,
                                             min_overlap_percentage=self.min_overlap, train=False,
                                             transform=self.test_image_transform)
        print('aaaaa')

    def train_dataloader(self):
        return DataLoader(self.ssspatch_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.ssspatch_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.ssspatch_test, batch_size=self.batch_size, num_workers=self.num_workers)
