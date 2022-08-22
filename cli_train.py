from pytorch_lightning.cli import LightningCLI

from data.ssspatch_datamodule import SSSPatchDataModule
from models.matching_train import MatchingTrain

# N.B the saved PL config is overwritten every time a new experiment is started!
# i.e. check out wandb/[run]/files/config.yaml for actual experiment config!
cli = LightningCLI(MatchingTrain, SSSPatchDataModule,
                   save_config_filename='config_cli.yaml',
                   save_config_overwrite=True,
                   env_parse=True)
