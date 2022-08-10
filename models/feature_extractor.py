import pytorch_lightning as pl
import torch
from torch import nn
from kornia import feature


class FeatureExtractor(pl.LightningModule):
    """Class used to extract various features from an input image"""

    def __init__(self, batch_size: int, num_kps: int, patch_size_for_feature: int = 32):
        self.patch_size_for_feature = patch_size_for_feature
        self.kps_to_local_affine_frames = lambda kps: feature.laf_from_center_scale_ori(
            torch.from_numpy(kps).reshape(batch_size, num_kps, 2)
        )

    def compute_features_using_kornia(self, image: torch.Tensor, kps: torch.Tensor, desc_algo: nn.Module,
                                      **kwargs) -> torch.Tensor:
        with torch.no_grad():
            local_affine_frames = self.kps_to_local_affine_frames(kps)
            features = feature.get_laf_descriptors(
                image,
                kps,
                patch_descriptor=desc_algo(**kwargs),
                patch_size=self.patch_size_for_feature
            )
            return features
