from typing import Optional, Any

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT

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

        trainer.logger.log_image(
            key='training_matches',
            images=[make_matching_plot_fast(batch['image0_norm'][0][0], batch['image1_norm'][0][0],
                                            batch['keypoints0'], batch['keypoints1'],
                                            batch['keypoints0'].permute(1, 2, 0).squeeze().numpy().reshape(-1, 2),
                                            batch['keypoints1'].permute(1, 2, 0).squeeze().numpy().reshape(-1, 2),
                                            # (0,1,0,1) = (r,g,b,a) = green
                                            color=np.tile([0, 1, 0, 1], (batch['keypoints0'].shape[1], 1)),
                                            text='GT'
                                            )],
            caption=[f'Ground truth matches patch ({batch["patch_id0"][0]}, {batch["patch_id1"][0]}']
        )