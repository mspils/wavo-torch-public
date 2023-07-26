"""
Main Module to start training
"""
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from lightning.pytorch.loggers import TensorBoardLogger

import metrics as mt
from callbacks import WaVoCallback
from data_tools import WaVoDataModule
from models import WaVoLightningModule

# print('set float matmul precision to high')
torch.set_float32_matmul_precision('high')


def main():
    """Main Method
    """

    config = {
        # 'filename': '../../data/input/Poetrau20.csv',
        # 'log_dir': '../../models_torch/pötrau',
        # 'level_name_org': 'WPoetrau_pegel_cm',
        'filename': '../../data/input/Treia.csv',
        'log_dir': '../../models_torch/treia',
        'level_name_org': 'Treia_pegel_cm',
        'train': 0.7,
        'val': 0.15,
        'test': 0.15,
        'batch_size': 2048,
        'in_size': 144,
        'out_size': 48,
        # 'scale_target':False,
        # 'differencing':1,
        'differencing': 0,
        'percentile': 0.95
    }

    found_lr = None
    tune_lr = False

    for _ in range(10):

        data_module = WaVoDataModule(**config)
        data_module.prepare_data()
        data_module.setup(stage='fit')

        # TODO kwargs for neural network?
        model = WaVoLightningModule(
            data_module.mean,
            data_module.scale,
            feature_count=data_module.feature_count,
            in_size=data_module.hparams.in_size,
            out_size=data_module.hparams.out_size,
            hidden_size_lstm=128,
            hidden_size=64,
            learning_rate=found_lr or 0.001)  # TODO tidy up scaler mess

        # Callbacks & Logging
        early_stop_callback = EarlyStopping(
            monitor="hp/val_loss", mode="min", patience=3)
        my_callback = WaVoCallback()
        callbacks = [early_stop_callback, my_callback]
        logger = TensorBoardLogger(config['log_dir'], default_hp_metric=False)

        trainer = pl.Trainer(default_root_dir=config['log_dir'],
                             logger=logger,
                             # accelerator="auto",
                             accelerator="gpu",
                             devices=1,
                             callbacks=callbacks,
                             max_epochs=50,
                             log_every_n_steps=1,
                             )

        if found_lr is None and tune_lr:
            tuner = pl.tuner.Tuner(trainer)
            # tuner.scale_batch_size(model, datamodule=data_module,init_val=512, max_trials=3)
            # TODO calling data_module here messes with the hyperparameter logging
            lr_finder = tuner.lr_find(model, datamodule=data_module)
            found_lr = lr_finder.suggestion()

        trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
