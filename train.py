from argparse import ArgumentParser

import matplotlib.pyplot as plt
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data.ssspatch_datamodule import SSSPatchDataModule
from models.logging_callbacks import LogImagesCallback
from models.matching_train import MatchingTrain

run_name = '220811_feat_extraction_test'
wandb_logger = WandbLogger(project='sss-corr', name=run_name, log_model='all')

parser = ArgumentParser()
parser = MatchingTrain.add_model_specific_args(parser)
parser = SSSPatchDataModule.add_model_specific_args(parser)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args(
    ['--descriptor_dim', '128',
     '--data_num_kps', '200',
     '--data_batch_size', '2',
     '--data_root',
     '/home/li/Documents/sss-correspondence/data/GullmarsfjordSMaRC20210209_ssh_annotations/survey2_better_resolution/9-0169to0182-nbr_pings-1301_annotated/patch240_step40_test0.1_refSSH-0170_OrderedDict/',
     '--data_num_workers', '0'])

ssspatch_sift_norm_img = SSSPatchDataModule(args)
ssspatch_sift_norm_img.setup()
print(f'Len train: {len(ssspatch_sift_norm_img.ssspatch_train)}')
print(f'Len val: {len(ssspatch_sift_norm_img.ssspatch_val)}')

model = MatchingTrain(args)
early_stopping_callback = EarlyStopping(monitor='val/loss', mode='min', patience=5, check_finite=True)
checkpoint_callback = ModelCheckpoint(monitor='val/loss',
                                      dirpath=f'/home/li/Documents/SuperGluePretrainedNetwork/{run_name}',
                                      filename='sss-epoch{epoch:02d}-val-loss-{val/loss:.2f}',
                                      save_top_k=3)
trainer = Trainer(logger=wandb_logger, max_epochs=2,
                  callbacks=[LogImagesCallback(),
                             early_stopping_callback],
                  accelerator='cpu',
                  devices=1,
                  gradient_clip_val=.5,
                  detect_anomaly=True)

# Train the model
trainer.fit(model, datamodule=ssspatch_sift_norm_img)