import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

NO_MATCH = -1


class SSSPatchDataset(Dataset):
    def __init__(self,
                 root: str,
                 min_overlap_percentage: float = .15,
                 train: bool = True,
                 transform=None,
                 num_kps: int = None):
        self.root = root
        self.train = train
        self.patch_root = os.path.join(self.root,
                                       'train' if self.train else 'test')
        self.npz_folder = os.path.join(self.patch_root, 'npz')
        self.num_kps = num_kps
        self.image_width, self.image_height = None, None

        self.min_overlap_percentage = min_overlap_percentage
        self.overlap_kps_dict = self._load_overlap_kps_dict()
        self.overlap_percentage_matrix, self.overlap_nbr_kps_matrix = self._load_overlap_matrices(
        )
        self.pairs_above_min_overlap_percentage = np.argwhere(
            self.overlap_percentage_matrix > self.min_overlap_percentage).astype(int)

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

    def _add_noisy_keypoints_and_modify_noisy_gt_match(self, data) -> dict:
        # Add noisy keypoints to image0 data
        keypoints0_new_indices = self._generate_random_idx_for_annotated_kps(data['keypoints0'].shape[0])
        keypoints0_noisy, gt_match0_noisy = self._add_noisy_keypoints(data['keypoints0'],
                                                                      data['gt_match0'],
                                                                      keypoints0_new_indices)

        # Add noisy keypoints to image1 data
        keypoints1_new_indices = self._generate_random_idx_for_annotated_kps(data['keypoints1'].shape[0])
        keypoints1_noisy, gt_match1_noisy = self._add_noisy_keypoints(data['keypoints1'],
                                                                      data['gt_match1'],
                                                                      keypoints1_new_indices)

        # Modify the indices in gt_match0_noisy and gt_match1_noisy to correspond to the noisy keypoint indices
        gt_match0_old_idx = gt_match0_noisy[keypoints0_new_indices]
        gt_match0_noisy[keypoints0_new_indices] = keypoints1_new_indices[gt_match0_old_idx]

        gt_match1_old_idx = gt_match1_noisy[keypoints1_new_indices]
        gt_match1_noisy[keypoints1_new_indices] = keypoints0_new_indices[gt_match1_old_idx]
        return {'noisy_keypoints0': keypoints0_noisy, 'noisy_keypoints1': keypoints1_noisy,
                'noisy_gt_match0': gt_match0_noisy,
                'noisy_gt_match1': gt_match1_noisy}

    def _add_noisy_keypoints(self, annotated_kps: np.array, gt_match: np.array,
                             annotated_kps_indices: np.array) -> dict:
        # TODO: switch between adding random keypoints, SIFT or other keypoints!

        # Generate random keypoints
        # TODO: get approx nadir from data
        approx_nadir = 150
        random_bin_nbr = np.random.randint(low=approx_nadir, high=self.image_width, size=self.num_kps)
        random_ping_nbr = np.random.randint(low=0, high=self.image_height, size=self.num_kps)
        noisy_kps = np.stack([random_bin_nbr, random_ping_nbr]).T

        # Place annotated_kps at randomly generated indices in the output noisy_kps array
        noisy_kps[annotated_kps_indices, :] = annotated_kps

        # Update gt_match to correspond to noisy_kps
        noisy_gt_match = np.ones(self.num_kps) * NO_MATCH
        noisy_gt_match[annotated_kps_indices] = gt_match
        return noisy_kps.astype(int), noisy_gt_match.astype(int)

    def _generate_random_idx_for_annotated_kps(self, num_annotated_kps: int) -> np.array:
        # TODO: handle the case where num_annotated_kps > self.num_kps (random sample up to self.num_kps?)
        return np.random.choice(self.num_kps, size=num_annotated_kps, replace=False).astype(int)

    def _load_patch(self, index: int):
        return np.load(os.path.join(self.npz_folder, f'{index}.npz'))

    def __len__(self) -> int:
        return self.pairs_above_min_overlap_percentage.shape[0]

    def __getitem__(self, index: int) -> dict:
        idx0, idx1 = self.pairs_above_min_overlap_percentage[index]
        print(f'index: {index}, idx0: {idx0}, idx1: {idx1}')
        patch0, patch1 = self._load_patch(idx0), self._load_patch(idx1)

        # Save image dimensions if None: assumes all images to be of the same dimensions
        if self.image_width is None:
            self.image_height, self.image_width = patch0['sss_waterfall_image'].shape

        raw_keypoints_and_matches = {'keypoints0': patch0['keypoints'], 'keypoints1': patch1['keypoints'],
                                     'gt_match0': np.array(self.overlap_kps_dict[str(idx0)][str(idx1)]),
                                     'gt_match1': np.array(self.overlap_kps_dict[str(idx1)][str(idx0)])}
        noisy_keypoints_and_matches = self._add_noisy_keypoints_and_modify_noisy_gt_match(raw_keypoints_and_matches)

        basic_patch_info_keys = ['idx', 'pos', 'rpy', 'sss_waterfall_image']
        raw_patch_info = {**{f'{k}0': patch0[k] for k in basic_patch_info_keys},
                          **{f'{k}1': patch1[k] for k in basic_patch_info_keys}
                          }
        processed_data = {**raw_patch_info, **noisy_keypoints_and_matches}
        data_torch = {k: torch.from_numpy(v).float() for k, v in processed_data.items()}
        # Modify image dimension: H x W -> 1 x H x W
        data_torch['sss_waterfall_image0'] = data_torch['sss_waterfall_image0'].unsqueeze(dim=0)
        data_torch['sss_waterfall_image1'] = data_torch['sss_waterfall_image1'].unsqueeze(dim=0)
        return data_torch


def get_matching_keypoints_according_to_matches(matches, keypoints0, keypoints1):
    """Given the proposed matches and two keypoint arrays, return two arrays of keypoints, where matching_kps0[i] is
    the corresponding keypoints to matching_kps1[i] according to the input matches array (which could be groundtruth
    or predictions)."""
    kps0_idx = torch.where(matches > NO_MATCH)
    matching_kps0 = keypoints0[kps0_idx]

    kps1_idx = matches[kps0_idx].to(int)
    # TODO: Handle batches > 1
    matching_kps1 = keypoints1[0, kps1_idx]
    return matching_kps0, matching_kps1
