from pytorch_lightning.cli import LightningCLI

from data.ssspatch_datamodule import SSSPatchDataModule
from models.matching_train import MatchingTrain

cli = LightningCLI(MatchingTrain, SSSPatchDataModule)
