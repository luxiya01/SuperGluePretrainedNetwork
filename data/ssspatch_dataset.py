import json
import pickle
import os
import numpy as np
from torch.utils.data import Dataset


class SSSPatchDataset(Dataset):
    def __init__(self,
                 root: str,
                 train: bool,
                 desc: str,
                 img_type: str,
                 min_overlap_percentage: float = .15,
                 transform=None):
        self.root = root
        self.train = train
        self.desc = desc
        self.img_type = img_type
        self.patch_root = os.path.join(self.root,
                                       'train' if self.train else 'test')
        self.desc_dir = os.path.join(self.patch_root, self.desc)

        self.min_overlap_percentage = min_overlap_percentage
        self.overlap_kps_dict = self._load_overlap_kps_dict()
        self.overlap_percentage_matrix, self.overlap_nbr_kps_matrix = self._load_overlap_matrices(
        )
        self.pairs_above_min_overlap_percentage = np.argwhere(
            self.overlap_percentage_matrix > self.min_overlap_percentage)

        self.transform = transform

    def _load_overlap_kps_dict(self) -> dict:
        filename = os.path.join(self.patch_root, 'overlap_kps.json')
        with open(filename, 'r', encoding='utf-8') as f:
            overlap_kps = json.load(f)
        return overlap_kps

    def _load_overlap_matrices(self) -> (np.ndarray, np.ndarray):
        filename = os.path.join(self.patch_root, 'overlap.npz')
        overlap_info = np.load(filename)
        overlap_percentage = np.triu(overlap_info['overlap_matrix'])
        overlap_nbr_kps = np.triu(overlap_info['overlap_nbr_kps'])
        return overlap_percentage, overlap_nbr_kps

    def __len__(self) -> int:
        return self.pairs_above_min_overlap_percentage.shape[0]

    def __getitem__(self, index: int) -> dict:
        idx0, idx1 = self.pairs_above_min_overlap_percentage[index]
        desc0, desc1 = self._load_desc(idx0), self._load_desc(idx1)
        data = {
            'keypoints0': desc0[f'kp_{self.img_type}'],
            'descriptors0': desc0[f'desc_{self.img_type}'],
            'keypoints1': desc1[f'kp_{self.img_type}'],
            'descriptors1': desc1[f'desc_{self.img_type}'],
            'groundtruth_match0':  self.overlap_kps_dict[str(idx0)][str(idx1)],
            'groundtruth_match1':  self.overlap_kps_dict[str(idx1)][str(idx0)],
        }
        return data

    def _load_desc(self, index: int):
        desc_path = os.path.join(self.desc_dir, f'patch{index}.npz')
        return np.load(desc_path)