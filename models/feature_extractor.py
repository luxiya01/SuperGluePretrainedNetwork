import pytorch_lightning as pl
import torch
from kornia import feature


class FeatureExtractor(pl.LightningModule):
    """Class used to extract various features from an input image at given keypoint locations."""

    def __init__(self, descriptor: pl.LightningModule, patch_size_for_feature: int):
        self.patch_size_for_feature = patch_size_for_feature
        self.descriptor = descriptor

    @staticmethod
    def kps_array_to_local_affine_frames(kps):
        batch_size, num_kps, _ = kps.shape
        return feature.laf_from_center_scale_ori(
            torch.from_numpy(kps).reshape(batch_size, num_kps, 2)
        )

    @torch.no_grad()
    def extract_features(self, image: torch.Tensor, kps: torch.Tensor) -> torch.Tensor:
        local_affine_frames = self.kps_to_local_affine_frames(kps)
        features = feature.get_laf_descriptors(
            image,
            lafs=local_affine_frames,
            patch_descriptor=self.descriptor,
            patch_size=self.patch_size_for_feature
        )
        return features
