from typing import Optional, Any

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT

from data.ssspatch_dataset import get_matching_keypoints_according_to_matches, NO_MATCH
from models.utils import make_matching_plot_fast


class LogImagesCallback(pl.Callback):
    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: Optional[STEP_OUTPUT],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        self._log_images(trainer, outputs, batch, stage='val')

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        self._log_images(trainer, outputs, batch, stage='train')

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._log_images(trainer, outputs, batch, stage='test')

    def _log_images(self, trainer: "pl.Trainer", outputs: Optional[STEP_OUTPUT], batch: Any, stage: str = 'val'):
        # Log GT matches
        gt_matching_kps0, gt_matching_kps1 = get_matching_keypoints_according_to_matches(
            matches=batch['groundtruth_match0'],
            keypoints0=batch['keypoints0'],
            keypoints1=batch['keypoints1'])

        # Log correct predictions
        correct_pred_matches0 = torch.where(batch['groundtruth_match0'] == outputs['pred']['matches0'],
                                            outputs['pred']['matches0'], NO_MATCH)
        correct_pred_kps0, correct_pred_kps1 = get_matching_keypoints_according_to_matches(
            matches=correct_pred_matches0,
            keypoints0=batch['keypoints0'],
            keypoints1=batch['keypoints1'])

        # Log incorrect predictions
        incorrect_pred_matches0 = torch.where(batch['groundtruth_match0'] != outputs['pred']['matches0'],
                                              outputs['pred']['matches0'], NO_MATCH)
        incorrect_pred_kps0, incorrect_pred_kps1 = get_matching_keypoints_according_to_matches(
            matches=incorrect_pred_matches0,
            keypoints0=batch['keypoints0'],
            keypoints1=batch['keypoints1'])

        matches = {'GT': {'mkpts0': gt_matching_kps0, 'mkpts1': gt_matching_kps1, 'color': [1, 1, 0, 1]},
                   'Incorrect predictions': {'mkpts0': incorrect_pred_kps0, 'mkpts1': incorrect_pred_kps1,
                                             'color': [1, 0, 0, 1]},
                   'Correct predictions': {'mkpts0': correct_pred_kps0, 'mkpts1': correct_pred_kps1,
                                           'color': [0, 1, 0, 1]}
                   }
        self._log_matches(trainer, batch, matches, stage=stage)

    @staticmethod
    def _log_matches(trainer: "pl.Trainer", batch: Any, matches: dict, stage: str = 'val'):
        trainer.logger.log_image(
            key=f'{stage}/matches',
            images=[make_matching_plot_fast(batch['image0_norm'][0][0].cpu(), batch['image1_norm'][0][0].cpu(),
                                            batch['keypoints0'][0].cpu().numpy(),
                                            batch['keypoints1'][0].cpu().numpy(),
                                            val['mkpts0'].cpu().numpy(),
                                            val['mkpts1'].cpu().numpy(),
                                            # color = (r,g,b,a)
                                            color=np.tile(val['color'], (batch['keypoints0'].shape[1], 1)),
                                            text=key,
                                            show_keypoints=True
                                            ) for key, val in matches.items()],
            caption=[
                f'{key} patches({batch["patch_id0"][0]}, {batch["patch_id1"][0]}) '
                f'nbr_matches = {len(val["mkpts0"])}' for key, val in matches.items()]
        )
