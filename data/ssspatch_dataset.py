import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode

NO_MATCH = -1


class SSSPatchDataset(Dataset):
    def __init__(self,
                 root: str,
                 desc: str,
                 img_type: str,
                 min_overlap_percentage: float = .15,
                 train: bool = True,
                 transform=None,
                 num_kps: int = None):
        self.root = root
        self.train = train
        self.desc = desc
        self.img_type = img_type
        self.patch_root = os.path.join(self.root,
                                       'train' if self.train else 'test')
        self.desc_dir = os.path.join(self.patch_root, self.desc)
        self.num_kps = num_kps
        self.image_width, self.image_height = None, None

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
        print(f'index: {index}, idx0: {idx0}, idx1: {idx1}')
        desc0, desc1 = self._load_desc(idx0), self._load_desc(idx1)
        data = {
            'keypoints0':
                desc0[f'kp_{self.img_type}'],
            'descriptors0':
                desc0[f'desc_{self.img_type}'].T,
            'scores0':
                np.ones(desc0[f'kp_{self.img_type}'].shape[0]),
            'keypoints1':
                desc1[f'kp_{self.img_type}'],
            'descriptors1':
                desc1[f'desc_{self.img_type}'].T,
            'scores1':
                np.ones(desc1[f'kp_{self.img_type}'].shape[0]),
            'groundtruth_match0':
                np.array(self.overlap_kps_dict[str(idx0)][str(idx1)]),
            'groundtruth_match1':
                np.array(self.overlap_kps_dict[str(idx1)][str(idx0)])
        }
        data_torch = {k: torch.from_numpy(v).float() for k, v in data.items()}
        data_torch['patch_id0'] = int(idx0)
        data_torch['patch_id1'] = int(idx1)
        data_torch['image0_raw'] = read_image(os.path.join(self.patch_root, f'patch{idx0}_intensity.png'),
                                              mode=ImageReadMode.RGB)
        data_torch['image0_norm'] = read_image(os.path.join(self.patch_root, f'patch{idx0}_norm_intensity.png'),
                                               mode=ImageReadMode.RGB)
        data_torch['image1_raw'] = read_image(os.path.join(self.patch_root, f'patch{idx1}_intensity.png'),
                                              mode=ImageReadMode.RGB)
        data_torch['image1_norm'] = read_image(os.path.join(self.patch_root, f'patch{idx1}_norm_intensity.png'),
                                               mode=ImageReadMode.RGB)
        if self.image_width is None:
            _, self.image_height, self.image_width = data_torch['image0_norm'].shape
        return data_torch

    def _load_desc(self, index: int):
        desc_path = os.path.join(self.desc_dir, f'patch{index}.npz')
        return np.load(desc_path)

    def _add_noisy_keypoints(self, annotated_kps: np.array, annotated_kps_indices: np.array) -> np.array:
        #TODO: switch between adding random keypoints, SIFT or other keypoints!

        # Generate random keypoints
        #TODO: get approx nadir from data
        approx_nadir = 150
        random_bin_nbr = np.random.randint(low=approx_nadir, high=self.image_width, size=self.num_output_kps)
        random_ping_nbr = np.random.randint(low=0, high=self.image_height, size=self.num_output_kps)
        noisy_kps = np.stack([random_bin_nbr, random_ping_nbr]).T

        # Place annotated_kps at randomly generated indices in the output noisy_kps array
        noisy_kps[annotated_kps_indices, :] = annotated_kps
        return noisy_kps

    def _generate_random_idx_for_annotated_kps(self, num_annotated_kps: int) -> np.array:
        return np.random.choice(self.num_output_kps, size=num_annotated_kps, replace=False)


def get_matching_keypoints_according_to_matches(matches, keypoints0, keypoints1):
    """Given the proposed matches and two keypoint arrays, return two arrays of keypoints, where matching_kps0[i] is
    the corresponding keypoints to matching_kps1[i] according to the input matches array (which could be groundtruth
    or predictions."""
    kps0_idx = torch.where(matches > NO_MATCH)
    matching_kps0 = keypoints0[kps0_idx]

    kps1_idx = matches[kps0_idx].to(int)
    # TODO: Handle batches > 1
    matching_kps1 = keypoints1[0, kps1_idx]
    return matching_kps0, matching_kps1
