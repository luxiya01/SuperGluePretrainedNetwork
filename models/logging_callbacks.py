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
        # Only log training images every n batch
        n = 50
        if batch_idx % n == 0:
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
        for i in range(batch['idx0'].shape[0]):
            # Log GT matches
            gt_matching_kps0, gt_matching_kps1 = get_matching_keypoints_according_to_matches(
                matches=batch['noisy_gt_match0'][i],
                keypoints0=batch['noisy_keypoints0'][i],
                keypoints1=batch['noisy_keypoints1'][i])

            # Log correct predictions
            correct_pred_matches0 = torch.where(batch['noisy_gt_match0'][i] == outputs['pred']['matches0'][i],
                                                outputs['pred']['matches0'][i], NO_MATCH)
            correct_pred_kps0, correct_pred_kps1 = get_matching_keypoints_according_to_matches(
                matches=correct_pred_matches0,
                keypoints0=batch['noisy_keypoints0'][i],
                keypoints1=batch['noisy_keypoints1'][i])

            # Log incorrect predictions
            incorrect_pred_matches0 = torch.where(batch['noisy_gt_match0'][i] != outputs['pred']['matches0'][i],
                                                  outputs['pred']['matches0'][i], NO_MATCH)
            incorrect_pred_kps0, incorrect_pred_kps1 = get_matching_keypoints_according_to_matches(
                matches=incorrect_pred_matches0,
                keypoints0=batch['noisy_keypoints0'][i],
                keypoints1=batch['noisy_keypoints1'][i])

            matches = {'GT': {'mkpts0': gt_matching_kps0, 'mkpts1': gt_matching_kps1, 'color': [0, 1, 1, 1]},
                       'Incorrect predictions': {'mkpts0': incorrect_pred_kps0, 'mkpts1': incorrect_pred_kps1,
                                                 'color': [0, 0, 1, 1]},
                       'Correct predictions': {'mkpts0': correct_pred_kps0, 'mkpts1': correct_pred_kps1,
                                               'color': [0, 1, 0, 1]}
                       }
            self._log_matches(trainer, batch, matches, idx_in_batch=i, stage=stage)

    @staticmethod
    def _log_matches(trainer: "pl.Trainer", batch: Any, matches: dict, idx_in_batch: int, stage: str = 'val'):
        trainer.logger.log_image(
            key=f'{stage}/matches',
            images=[make_matching_plot_fast(batch['image0'][idx_in_batch][0].cpu().numpy(),
                                            batch['image1'][idx_in_batch][0].cpu().numpy(),
                                            batch['noisy_keypoints0'][idx_in_batch].cpu().numpy(),
                                            batch['noisy_keypoints1'][idx_in_batch].cpu().numpy(),
                                            val['mkpts0'].cpu().numpy(),
                                            val['mkpts1'].cpu().numpy(),
                                            # color = (r,g,b,a)
                                            color=np.tile(val['color'], (val['mkpts0'].shape[0], 1)),
                                            text=key,
                                            show_keypoints=True,
                                            is_noisy_kpts0=batch['is_noisy_keypoints0'][idx_in_batch].cpu().numpy(),
                                            is_noisy_kpts1=batch['is_noisy_keypoints1'][idx_in_batch].cpu().numpy()
                                            ) for key, val in matches.items()],
            caption=[
                f'{key} patches({batch["idx0"][idx_in_batch]}, {batch["idx1"][idx_in_batch]}) '
                f'nbr_matches = {len(val["mkpts0"])}' for key, val in matches.items()]
        )
