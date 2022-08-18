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

    # TODO: allow gradient to flow through the feature extraction modules...!?
    @torch.no_grad()
    def extract_features(self, image: torch.Tensor, kps: torch.Tensor) -> torch.Tensor:
        """Returns features with shape (batch_size, num_kps, descriptor_size)"""
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
    def __init__(self, patch_size: int = 32, num_ang_bins: int = 8, num_spatial_bins: int = 4, rootsift: bool = False,
                 clipval: float = .2):
        """Output feature size = num_ang_bins * (num_spatial_bins^2)"""
        super(SIFTFeatureExtractor, self).__init__(patch_size_for_feature=patch_size)
        self.output_dims = num_ang_bins * (num_spatial_bins ** 2)
        self.descriptor = feature.SIFTDescriptor(
            patch_size=patch_size,
            num_ang_bins=num_ang_bins,
            num_spatial_bins=num_spatial_bins,
            rootsift=rootsift,
            clipval=clipval
        )


class HardNetFeatureExtractor(FeatureExtractor):
    def __init__(self, pretrained: bool = True):
        """Output feature size fixed to = 128. Can only compute patch_size of 32x32."""
        super(HardNetFeatureExtractor, self).__init__(patch_size_for_feature=32)
        self.output_dims = 128
        self.descriptor = feature.HardNet(pretrained=pretrained)


class HardNet8FeatureExtractor(FeatureExtractor):
    def __init__(self, pretrained: bool = True):
        """Output feature size fixed to = 128. Can only compute patch_size of 32x32."""
        super(HardNet8FeatureExtractor, self).__init__(patch_size_for_feature=32)
        self.output_dims = 128
        self.descriptor = feature.HardNet8(pretrained=pretrained)


class MKDDFeatureExtractor(FeatureExtractor):
    def __init__(self, patch_size: int = 32, kernel_type: str = 'concat',
                 whitening: str = 'pcawt', training_set='liberty', output_dims: int = 128):
        super(MKDDFeatureExtractor, self).__init__(patch_size_for_feature=patch_size)
        self.output_dims = output_dims
        self.descriptor = feature.MKDDescriptor(patch_size=patch_size,
                                                kernel_type=kernel_type,
                                                whitening=whitening,
                                                training_set=training_set,
                                                output_dims=output_dims)


class HyNetFeatureExtractor(FeatureExtractor):
    def __init__(self, pretrained: bool = True, is_bias: bool = True,
                 is_bias_FRN: bool = True, output_dims: int = 128, drop_rate: float = .3):
        """Can only compute patch_size of 32x32."""
        super(HyNetFeatureExtractor, self).__init__(patch_size_for_feature=32)
        self.output_dims = output_dims
        self.descriptor = feature.HyNet(pretrained=pretrained,
                                        is_bias=is_bias,
                                        is_bias_FRN=is_bias_FRN,
                                        dim_desc=output_dims,
                                        drop_rate=drop_rate)


class TFeatFeatureExtractor(FeatureExtractor):
    def __init__(self, pretrained: bool = True):
        """Output feature size fixed to = 128. Can only compute patch_size of 32x32."""
        super(TFeatFeatureExtractor, self).__init__(patch_size_for_feature=32)
        self.output_dims = 128
        self.descriptor = feature.TFeat(pretrained=pretrained)


class SOSNetFeatureExtractor(FeatureExtractor):
    def __init__(self, pretrained: bool = True):
        """Output feature size fixed to = 128. Can only compute patch_size of 32x32."""
        super(SOSNetFeatureExtractor, self).__init__(patch_size_for_feature=32)
        self.output_dims = 128
        self.descriptor = feature.SOSNet(pretrained=pretrained)
