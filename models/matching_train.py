import numpy as np
import pytorch_lightning as pl
import torch

from models.utils import make_matching_plot_fast
from .superglue import SuperGlue


class MatchingTrain(pl.LightningModule):
    """ Image Matching Frontend (SuperGlue) """

    # TODO: Add metrics such as TP, FP, precision, recall to training, validation and testing steps!
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.superglue = SuperGlue(config)

        self.save_hyperparameters()
        print(self.hparams)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MatchingTrain")
        parser.add_argument('--descriptor_dim', type=int, default=256)
        parser.add_argument('--keypoint_encoder', type=list, default=[32, 64, 128, 256])
        parser.add_argument('--gnn_layers', type=list, default=['self', 'cross'] * 9)
        parser.add_argument('--sinkhorn_iterations', type=int, default=100)
        parser.add_argument('--match_threshold', type=float, default=0.2)
        return parent_parser

    def forward(self, data):
        """ Run SuperGlue on input data
        Args:
          data: dictionary with keys: [
          'keypoints0', 'keypoints1', 'descriptors0', 'descriptors1',
          'scores0', 'scores1']
        """

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        pred = {**self.superglue(data), **data}
        return pred

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch)
        pred_scores = pred['scores']
        loss = self.compute_loss(pred_scores, batch)

        self.log('train/loss', loss, on_step=True, on_epoch=True)

        self.logger.log_image(
            key='training_images',
            images=[batch['image0_norm'].to(torch.float), batch['image1_norm'].to(torch.float)],
            caption=[f'patch {batch["patch_id0"][0]}', f'patch {batch["patch_id1"][0]}']
        )

        self.logger.log_image(
            key='training_matches',
            images=[make_matching_plot_fast(batch['image0_norm'][0][0], batch['image1_norm'][0][0],
                                            batch['keypoints0'], batch['keypoints1'],
                                            batch['keypoints0'].permute(1, 2, 0).squeeze().numpy().reshape(-1, 2),
                                            batch['keypoints1'].permute(1, 2, 0).squeeze().numpy().reshape(-1, 2),
                                            # (0,1,0,1) = (r,g,b,a) = green
                                            color=np.tile([0, 1, 0, 1], (batch['keypoints0'].shape[1], 1)),
                                            text='GT'
                                            )])

        # self.log('train/pred', pred, on_step=True, on_epoch=True)
        return {'loss': loss, 'pred': pred}

    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch)
        pred_scores = pred['scores']
        loss = self.compute_loss(pred_scores, batch)

        self.log('val/loss', loss, on_step=False, on_epoch=True)
        # self.log('val/pred', pred, on_step=False, on_epoch=True)
        return {'loss': loss, 'pred': pred}

    def test_step(self, batch):
        pred = self.forward(batch)
        pred_scores = pred['scores']
        loss = self.compute_loss(pred_scores, batch)

        self.log('test/loss', loss, on_step=False, on_epoch=True)
        # self.log('test/pred', pred, on_step=False, on_epoch=True)
        return {'loss': loss, 'pred': pred}

    def compute_loss(self, pred_scores, batch):
        gt0_with_matches, gt0_no_matches = gt_to_array_with_and_without_matches(
            batch['groundtruth_match0'])
        gt1_with_matches, gt1_no_matches = gt_to_array_with_and_without_matches(
            batch['groundtruth_match1'])

        # loss from GT matches
        loss = -pred_scores[:, gt0_with_matches[:, 0],
                gt0_with_matches[:, 1]].sum()
        # loss from kps from image0 without matches (matches to dust_bin with col idx = -1)
        loss -= pred_scores[:, gt0_no_matches[:, 0], gt0_no_matches[:,
                                                     1]].sum()
        # loss from kps from image1 without matches (matches to dust_bin with row idx = -1)
        loss -= pred_scores[:, gt1_no_matches[:, 1], gt1_no_matches[:,
                                                     0]].sum()
        nbr_matches = gt0_with_matches.shape[0] + gt0_no_matches.shape[
            0] + gt1_no_matches.shape[0]
        loss = loss / nbr_matches
        return loss

    #   def validation_epoch_end(self, validation_step_outputs):
    #       #TODO: write validation epoch: save the model
    #       flattened_predictions = torch.flatten(
    #           torch.cat(validation_step_outputs['pred']))
    #       self.logger.experiment.log({
    #           'valid/predictions':
    #           wandb.Histogram(flattened_predictions.to('cpu')),
    #           'global_step':
    #           self.global_step
    #       })

    #   def test_epoch_end(self, test_step_outputs):
    #       #TODO: write test epoch end: save the model!
    #       return super().test_epoch_end(test_step_outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


def gt_to_array_with_and_without_matches(gt_list):
    # convert gt list to a torch tensor with shape (batch_size, num_kps, 2)
    # and then split the array into two based on whether the kp has GT correspondence
    # the returned two arrays both have shape (num_kps, 2)

    # TODO: handle batch_size > 1
    gt_array = torch.stack(
        (torch.arange(gt_list.shape[1]).unsqueeze(0), torch.tensor(gt_list)),
        axis=2).long()
    gt_with_matches = gt_array[torch.where(gt_array[:, :, 1] != -1)]
    gt_without_matches = gt_array[torch.where(gt_array[:, :, 1] == -1)]
    return gt_with_matches, gt_without_matches
