import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from kornia import feature
from torchvision.utils import make_grid
import torchvision.transforms.functional as F


class FeatureExtractor(pl.LightningModule):
    """Class used to extract various features from an input image at given keypoint locations."""

    def __init__(self, descriptor: pl.LightningModule = None, patch_size_for_feature: int = 32):
        super(FeatureExtractor, self).__init__()
        self.patch_size_for_feature = patch_size_for_feature
        self.descriptor = descriptor

    @staticmethod
    def kps_array_to_local_affine_frames(kps):
        if isinstance(kps, np.ndarray):
            batch_size, num_kps, _ = kps.shape
            kps = torch.from_numpy(kps).reshape(batch_size, num_kps, 2)
        return feature.laf_from_center_scale_ori(kps)

    @torch.no_grad()
    def extract_features(self, image: torch.Tensor, kps: torch.Tensor) -> torch.Tensor:
        local_affine_frames = self.kps_array_to_local_affine_frames(kps)
        features = feature.get_laf_descriptors(
            image,
            lafs=local_affine_frames,
            patch_descriptor=self.descriptor,
            patch_size=self.patch_size_for_feature
        )
        return features

    @staticmethod
    def show_features(features):
        grid = make_grid(features)

        fig, ax = plt.subplots(ncols=len(grid), squeeze=False)
        for i, feat in enumerate(grid):
            feat = feat.detach()
            feat = F.to_pil_image(feat)
            ax[0, i].imshow(np.asarray(feat))


class SIFTFeatureExtractor(FeatureExtractor):
    def __init__(self, patch_size: int = 41, num_ang_bins: int = 8, num_spatial_bins: int = 4, rootsift: bool = False,
                 clipval: float = .2):
        """Output feature size = num_ang_bins * (num_spatial_bins^2)"""
        super(SIFTFeatureExtractor, self).__init__(patch_size_for_feature=patch_size)
        self.descriptor = feature.SIFTDescriptor(
            patch_size=patch_size,
            num_ang_bins=num_ang_bins,
            num_spatial_bins=num_spatial_bins,
            rootsift=rootsift,
            clipval=clipval
        )


class HardNetFeatureExtractor(FeatureExtractor):
    def __init__(self, pretrained: bool = False):
        """Output feature size fixed to = 128"""
        super(HardNetFeatureExtractor, self).__init__(patch_size_for_feature=32)
        self.descriptor = feature.HardNet(pretrained=pretrained)
