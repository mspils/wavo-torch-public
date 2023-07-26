"""
Main Module to start training
"""
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.cli import LightningCLI

from lightning.pytorch.loggers import TensorBoardLogger

import metrics as mt
from callbacks import WaVoCallback
from data_tools import WaVoDataModule
from models import WaVoLightningModule

# print('set float matmul precision to high')
torch.set_float32_matmul_precision('high')


class MyLightningCLI(LightningCLI):
    # def __init__():
    #    super().__init__()

    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.mean", "model.mean",
                              apply_on="instantiate")
        parser.link_arguments("data.scale", "model.scale",
                              apply_on="instantiate")
        parser.link_arguments(
            "data.threshold", "model.threshold", apply_on="instantiate")
        parser.link_arguments("data.feature_count",
                              "model.feature_count", apply_on="instantiate")
        parser.link_arguments("data.in_size", "model.in_size")
        parser.link_arguments("data.out_size", "model.out_size")

        parser.add_lightning_class_args(WaVoCallback, "wavo_callback")
        # parser.set_defaults({"wavo_callback.chosen_metrics": ['nse', 'p10']})

        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        parser.set_defaults({"early_stopping.monitor": "hp/val_loss",
                            'early_stopping.mode': 'min', "early_stopping.patience": 2})
        # monitor="hp/val_loss", mode="min"


if __name__ == "__main__":

    cli = MyLightningCLI(model_class=WaVoLightningModule,
                         datamodule_class=WaVoDataModuleCLI,
                         )
    # run=False
    # hidden_size_lstm: 128
    # hidden_size: 64
    # num_layers: 2
    # dropout: 0.2
    # learning_rate: 0.001
    # lightning_model = WaVoLightningModule(
    #    cli.data_module.mean,
    #    cli.data_module.scale,
    #    cli.data_module.threshold,
    #    feature_count=cli.data_module.feature_count,
    #    in_size=cli.data_module.hparams.in_size,
    #    out_size=cli.data_module.hparams.out_size,
    #    hidden_size_lstm=cli.model.hidden_size_lstm,
    #    hidden_size=cli.model.hidden_size,
    #    learning_rate=cli.model.learning_rate)
    # cli.trainer.fit(lightning_model, datamodule=cli.datamodule)
