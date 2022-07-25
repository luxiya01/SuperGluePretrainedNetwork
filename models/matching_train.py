import pytorch_lightning as pl
import torch

from data.ssspatch_dataset import NO_MATCH
from .superglue import SuperGlue


class MatchingTrain(pl.LightningModule):
    """ Image Matching Frontend (SuperGlue) """

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
        metrics = self.compute_metrics(pred)

        self.log('train/loss', loss)
        for k, v in metrics.items():
            self.log(f'train/{k}', v)
        return {'loss': loss, 'pred': pred, **metrics}

    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch)
        pred_scores = pred['scores']
        loss = self.compute_loss(pred_scores, batch)
        metrics = self.compute_metrics(pred)

        self.log('val/loss', loss)
        for k, v in metrics.items():
            self.log(f'val/{k}', v)
        return {'loss': loss, 'pred': pred, **metrics}

    def test_step(self, batch):
        pred = self.forward(batch)
        pred_scores = pred['scores']
        loss = self.compute_loss(pred_scores, batch)
        metrics = self.compute_metrics(pred)

        self.log('test/loss', loss)
        self.log_dict(metrics)
        for k, v in metrics.items():
            self.log(f'test/{k}', v)
        return {'loss': loss, 'pred': pred, **metrics}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def gt_to_array_with_and_without_matches(self, gt_list):
        # convert gt list to a torch tensor with shape (batch_size, num_kps, 2)
        # and then split the array into two based on whether the kp has GT correspondence
        # the returned two arrays both have shape (num_kps, 2)

        # TODO: handle batch_size > 1
        gt_array = torch.stack(
            (torch.arange(gt_list.shape[1]).unsqueeze(0).to(self.device), gt_list.clone().detach()),
            axis=2).long()
        gt_with_matches = gt_array[torch.where(gt_array[:, :, 1] != -1)]
        gt_without_matches = gt_array[torch.where(gt_array[:, :, 1] == -1)]
        return gt_with_matches, gt_without_matches

    def compute_loss(self, pred_scores, batch):
        gt0_with_matches, gt0_no_matches = self.gt_to_array_with_and_without_matches(
            batch['groundtruth_match0'])
        gt1_with_matches, gt1_no_matches = self.gt_to_array_with_and_without_matches(
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

    @staticmethod
    def compute_metrics(pred):
        predictions = torch.concat([pred['matches0'], pred['matches1']], dim=1).flatten()
        gt = torch.concat([pred['groundtruth_match0'], pred['groundtruth_match1']], dim=1).flatten()

        num_predicted_matches = torch.count_nonzero(predictions != NO_MATCH)
        num_gt_matches = torch.count_nonzero(gt != NO_MATCH)
        num_correctly_predicted_matches = torch.count_nonzero(torch.logical_and(predictions == gt, gt != NO_MATCH))

        precision = num_correctly_predicted_matches / num_predicted_matches if num_predicted_matches > 0 else 0
        recall = num_correctly_predicted_matches / num_gt_matches if num_gt_matches > 0 else 0
        matching_score = num_correctly_predicted_matches / predictions.shape[0]
        accuracy = torch.count_nonzero(predictions == gt) / predictions.shape[0]

        return {'precision': precision, 'recall': recall, 'matching_score': matching_score, 'accuracy': accuracy,
                'TP': num_correctly_predicted_matches, 'FP': num_predicted_matches - num_correctly_predicted_matches}
