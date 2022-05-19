import torch
import pytorch_lightning as pl
import numpy as np

from .superpoint import SuperPoint
from .superglue import SuperGlue


class MatchingTrain(pl.LightningModule):
    """ Image Matching Frontend (SuperGlue) """
    def __init__(self, config={}):
        super().__init__()
        self.superglue = SuperGlue(config.get('superglue', {}))
        self.loss = NLLLoss()

    def forward(self, data):
        """ Run SuperGlue on input data
        Args:
          data: dictionary with keys: ['image0', 'image1',
          'keypoints0', 'keypoints1', 'descriptors0', 'descriptors1',
          'scores0', 'scores1']
        """

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        pred = {**self.superglue(data)}
        return pred

    def training_step(self, batch, batch_idx):
        data, gt = batch
        pred = self.forward(data)
        pred_scores = pred['scores']

        gt0_with_matches, gt0_no_matches = gt_to_array_with_and_without_matches(gt[0])
        gt1_with_matches, gt1_no_matches = gt_to_array_with_and_without_matches(gt[1])

        # loss from GT matches
        loss = -pred_scores[gt0_with_matches[:, 0], gt0_with_matches[:, 1]]
        # loss from kps from image0 without matches (matches to dust_bin with col idx = -1)
        loss -= pred_scores[gt0_no_matches[:, 0], gt0_no_matches[:, 1]]
        # loss from kps from image1 without matches (matches to dust_bin with row idx = -1)
        loss -= pred_scores[gt1_no_matches[:, 1], gt1_no_matches[:, 0]]
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


def gt_to_array_with_and_without_matches(gt_list):
    # convert gt list to a torch tensor with shape (num_kps, 2)
    # and then split the array into two based on whether the kp has GT correspondence
    gt_array = torch.stack((torch.arange(len(gt_list)), torch.tensor(gt_list)), axis=1)
    gt_with_matches = gt_array[torch.where(gt_array[:, 1] != -1)]
    gt_without_matches = gt_array[torch.where(gt_array[:, 1] == -1)]
    return gt_with_matches, gt_without_matches