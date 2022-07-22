from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from data.ssspatch_datamodule import SSSPatchDataModule
from models.matching_train import MatchingTrain

from models.logging_callbacks import LogImagesCallback

wandb_logger = WandbLogger(project='sss-corr', name='220722_log_predictions')

parser = ArgumentParser()
parser = MatchingTrain.add_model_specific_args(parser)
parser = SSSPatchDataModule.add_model_specific_args(parser)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args(
    ['--descriptor_dim', '128',
     '--data_root',
     '/home/li/Documents/sss-correspondence/data/GullmarsfjordSMaRC20210209_ssh_annotations/survey2_better_resolution/9-0169to0182-nbr_pings-1301_annotated/patch240_step40_test0.1_refSSH-0170_OrderedDict/',
     '--data_num_workers', '0'])

model = MatchingTrain(args)
# TODO: use GPU
# trainer = Trainer(logger=wandb_logger, accelerator='gpu', devices=1, max_epochs=10)
trainer = Trainer(logger=wandb_logger, max_epochs=1, callbacks=[LogImagesCallback()])

ssspatch_sift_norm_img = SSSPatchDataModule(args)
ssspatch_sift_norm_img.setup()
print(f'Len train: {len(ssspatch_sift_norm_img.ssspatch_train)}')
print(f'Len val: {len(ssspatch_sift_norm_img.ssspatch_val)}')

# Train the model
trainer.fit(model, datamodule=ssspatch_sift_norm_img)
