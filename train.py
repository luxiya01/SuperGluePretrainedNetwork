from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from models.matching_train import MatchingTrain
from data.ssspatch_datamodule import SSSPatchDataModule

wandb_logger = WandbLogger(project='sss-corr', name='220715_log_test')
model = MatchingTrain(config={'descriptor_dim': 128,
                              'keypoint_encoder': [32, 64, 128]})
# TODO: use GPU
# trainer = Trainer(logger=wandb_logger, accelerator='gpu', devices=1, max_epochs=10)
trainer = Trainer(logger=wandb_logger, max_epochs=1)

ssspatch_sift_norm_img = SSSPatchDataModule(
    root=
    '/home/li/Documents/sss-correspondence/data/GullmarsfjordSMaRC20210209_ssh_annotations/survey2_better_resolution'
    '/9-0169to0182-nbr_pings-1301_annotated/patch240_step40_test0.1_refSSH-0170/',
    desc='sift',
    img_type='norm',
    min_overlap_percentage=0.15,
    eval_split=.1,
    batch_size=1,
    num_workers=0)
ssspatch_sift_norm_img.setup()
print(f'Len train: {len(ssspatch_sift_norm_img.ssspatch_train)}')
print(f'Len val: {len(ssspatch_sift_norm_img.ssspatch_val)}')

# Train the model
trainer.fit(model, datamodule=ssspatch_sift_norm_img)