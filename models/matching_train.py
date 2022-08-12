import pytorch_lightning as pl
import torch

from data.ssspatch_dataset import NO_MATCH
from .feature_extractor import SIFTFeatureExtractor
from .superglue import SuperGlue


class MatchingTrain(pl.LightningModule):
    """ Image Matching Frontend (various descriptors) + Backend (SuperGlue) """

    def __init__(self, config):
        super().__init__()
        self.config = config
        # TODO: allow changing descriptors
        self.descriptor = SIFTFeatureExtractor(patch_size=32)
        self.superglue = SuperGlue(config)
        self.learning_rate = config.learning_rate

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
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        return parent_parser

    def forward(self, data):
        """ Run SuperGlue on input data
        Args:
          data: dictionary with keys: [
          'idx0', 'sss_waterfall_image0', 'noisy_keypoints0', 'noisy_gt_match0',
          'idx1', 'sss_waterfall_image1', 'noisy_keypoints1', 'noisy_gt_match1']
        """

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])
        data['descriptors0'] = self.descriptor.extract_features(data['sss_waterfall_image0'], data[
            'noisy_keypoints0']).transpose(1, 2)
        data['descriptors1'] = self.descriptor.extract_features(data['sss_waterfall_image1'], data[
            'noisy_keypoints1']).transpose(1, 2)

        pred = {**self.superglue(data), **data}
        return pred

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch)
        pred_scores = pred['scores']
        loss = self.compute_loss(pred_scores, batch)
        metrics = self.compute_metrics(pred)

        self.log('train/loss', loss, on_step=True, on_epoch=True)
        for k, v in metrics.items():
            self.log(f'train/{k}', v, on_step=True, on_epoch=True)
        return {'loss': loss, 'pred': pred, **metrics}

    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch)
        pred_scores = pred['scores']
        loss = self.compute_loss(pred_scores, batch)
        metrics = self.compute_metrics(pred)

        self.log('val/loss', loss, on_step=True, on_epoch=True)
        for k, v in metrics.items():
            self.log(f'val/{k}', v, on_step=True, on_epoch=True)
        return {'loss': loss, 'pred': pred, **metrics}

    def test_step(self, batch):
        pred = self.forward(batch)
        pred_scores = pred['scores']
        loss = self.compute_loss(pred_scores, batch)
        metrics = self.compute_metrics(pred)

        self.log('test/loss', loss, on_step=True, on_epoch=True)
        for k, v in metrics.items():
            self.log(f'test/{k}', v, on_step=True, on_epoch=True)
        return {'loss': loss, 'pred': pred, **metrics}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def compute_loss(self, pred_scores, batch):
        batch_gt0_with_match, kps_gt0_with_match = torch.where(batch['noisy_gt_match0'] != NO_MATCH)
        corresponding_gt1_match = batch['noisy_gt_match0'][batch_gt0_with_match, kps_gt0_with_match].to(int)

        batch_gt0_no_match, kps_gt0_no_match = torch.where(batch['noisy_gt_match0'] == NO_MATCH)
        batch_gt1_no_match, kps_gt1_no_match = torch.where(batch['noisy_gt_match1'] == NO_MATCH)

        # loss from GT matches
        pred_score_elems_with_match = pred_scores[batch_gt0_with_match, kps_gt0_with_match, corresponding_gt1_match]
        _, counts = torch.unique(batch_gt0_with_match, return_counts=True)
        weights = 1/counts[batch_gt0_with_match]
        loss = (weights*pred_score_elems_with_match).sum()

        # loss from kps from image0 without matches (matches to dust_bin with col idx = -1)
        _, counts = torch.unique(batch_gt0_no_match, return_counts=True)
        weights = 1/counts[batch_gt0_no_match]
        loss -= (weights*pred_scores[batch_gt0_no_match, kps_gt0_no_match, -1]).sum()

        # loss from kps from image1 without matches (matches to dust_bin with row idx = -1)
        _, counts = torch.unique(batch_gt1_no_match, return_counts=True)
        weights = 1/counts[batch_gt1_no_match]
        loss -= (weights*pred_scores[batch_gt1_no_match, -1, kps_gt1_no_match]).sum()

        batch_size = pred_scores.shape[0]
        return loss/batch_size

    @staticmethod
    def compute_metrics(pred):
        #TODO: update this function for larger batches!
        predictions = torch.concat([pred['matches0'], pred['matches1']], dim=1).flatten()
        gt = torch.concat([pred['gt_match0'], pred['gt_match1']], dim=1).flatten()

        num_predicted_matches = torch.count_nonzero(predictions != NO_MATCH)
        num_gt_matches = torch.count_nonzero(gt != NO_MATCH)
        num_correctly_predicted_matches = torch.count_nonzero(torch.logical_and(predictions == gt, gt != NO_MATCH))

        precision = num_correctly_predicted_matches / num_predicted_matches if num_predicted_matches > 0 else 0
        recall = num_correctly_predicted_matches / num_gt_matches if num_gt_matches > 0 else 0
        matching_score = num_correctly_predicted_matches / predictions.shape[0]
        accuracy = torch.count_nonzero(predictions == gt) / predictions.shape[0]

        return {'precision': precision, 'recall': recall, 'matching_score': matching_score, 'accuracy': accuracy,
                'TP': num_correctly_predicted_matches, 'FP': num_predicted_matches - num_correctly_predicted_matches}
