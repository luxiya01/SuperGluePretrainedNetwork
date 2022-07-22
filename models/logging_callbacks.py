from typing import Optional, Any, List

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
        # Log patches
        trainer.logger.log_image(
            key='val/images',
            images=[batch['image0_norm'].to(torch.float), batch['image1_norm'].to(torch.float)],
            caption=[f'patch {batch["patch_id0"][0]}', f'patch {batch["patch_id1"][0]}']
        )

        # Log GT matches
        matching_kps0, matching_kps1 = get_matching_keypoints_according_to_matches(matches=batch['groundtruth_match0'],
                                                                                   keypoints0=batch['keypoints0'],
                                                                                   keypoints1=batch['keypoints1'])
        self._log_matches(trainer, batch, matching_kps0, matching_kps1, match_type='GT', stage='val',
                          color=[1, 1, 0, 1])

        # Log correct predictions
        correct_pred_matches0 = torch.where(batch['groundtruth_match0'] == outputs['pred']['matches0'],
                                            outputs['pred']['matches0'], NO_MATCH)
        correct_pred_kps0, correct_pred_kps1 = get_matching_keypoints_according_to_matches(
            matches=correct_pred_matches0,
            keypoints0=batch['keypoints0'],
            keypoints1=batch['keypoints1'])
        self._log_matches(trainer, batch, correct_pred_kps0, correct_pred_kps1, match_type='Correct pred', stage='val',
                          color=[0, 1, 0, 1])

        # Log incorrect predictions
        incorrect_pred_matches0 = torch.where(batch['groundtruth_match0'] != outputs['pred']['matches0'],
                                              outputs['pred']['matches0'], NO_MATCH)
        incorrect_pred_kps0, incorrect_pred_kps1 = get_matching_keypoints_according_to_matches(
            matches=incorrect_pred_matches0,
            keypoints0=batch['keypoints0'],
            keypoints1=batch['keypoints1'])
        self._log_matches(trainer, batch, incorrect_pred_kps0, incorrect_pred_kps1, match_type='Incorrect pred',
                          stage='val',
                          color=[1, 0, 0, 1])

    def _log_matches(self, trainer: "pl.Trainer", batch: Any, matching_kps0: torch.Tensor, matching_kps1: torch.Tensor,
                     match_type: str, stage: str = 'val', color: List = [0, 1, 0, 1]):
        trainer.logger.log_image(
            key=f'{stage}/matches/{match_type}',
            images=[make_matching_plot_fast(batch['image0_norm'][0][0], batch['image1_norm'][0][0],
                                            batch['keypoints0'][0].numpy(),
                                            batch['keypoints1'][0].numpy(),
                                            matching_kps0.numpy(),
                                            matching_kps1.numpy(),
                                            # color = (r,g,b,a)
                                            color=np.tile(color, (batch['keypoints0'].shape[1], 1)),
                                            text='GT',
                                            show_keypoints=True
                                            )],
            caption=[
                f'{match_type} matches patches({batch["patch_id0"][0]}, {batch["patch_id1"][0]})'
                f'nbr_matches = {len(matching_kps1)}']
        )
