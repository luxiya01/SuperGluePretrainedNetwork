from typing import Optional, Any

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT

from data.ssspatch_dataset import get_groundtruth_matching_keypoints
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
        trainer.logger.log_image(
            key='images',
            images=[batch['image0_norm'].to(torch.float), batch['image1_norm'].to(torch.float)],
            caption=[f'patch {batch["patch_id0"][0]}', f'patch {batch["patch_id1"][0]}']
        )

        matching_kps0, matching_kps1 = get_groundtruth_matching_keypoints(batch)
        trainer.logger.log_image(
            key='training_matches',
            # TO
            images=[make_matching_plot_fast(batch['image0_norm'][0][0], batch['image1_norm'][0][0],
                                            batch['keypoints0'][0].numpy(),
                                            batch['keypoints1'][0].numpy(),
                                            matching_kps0.numpy(),
                                            matching_kps1.numpy(),
                                            # batch['keypoints0'].permute(1, 2, 0).squeeze().numpy().reshape(-1, 2),
                                            # batch['keypoints1'].permute(1, 2, 0).squeeze().numpy().reshape(-1, 2),
                                            # (0,1,0,1) = (r,g,b,a) = green
                                            color=np.tile([0, 1, 0, 1], (batch['keypoints0'].shape[1], 1)),
                                            text='GT',
                                            show_keypoints=True
                                            )],
            caption=[f'Ground truth matches patch ({batch["patch_id0"][0]}, {batch["patch_id1"][0]}']
        )
