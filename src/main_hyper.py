"""
Main Module to start training
"""
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.cli import LightningCLI

from lightning.pytorch.loggers import TensorBoardLogger

import metrics as mt
from callbacks import WaVoCallBack
from data_tools import WaVoDataModule
from models import WaVoLightningModule

import optuna
from optuna.integration import PyTorchLightningPruningCallback

# print('set float matmul precision to high')
torch.set_float32_matmul_precision('high')


def main():
    """Main Method
    """

    config = {'filename': '../../data/input/Poetrau20.csv',
              'log_dir': '../../models_torch/pötrau',
              'level_name_org': 'WPoetrau_pegel_cm',
              # config = {'filename': '../../data/input/Treia.csv',
              #          'log_dir': '../../models_torch/treia',
              #          'level_name_org': 'Treia_pegel_cm',
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

        # data_module = WaVoDataModule(**config_debug)
        data_module = WaVoDataModule(**config)
        data_module.prepare_data()
        data_module.setup(stage='fit')

        # print(data_module.scaler)
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
        my_callback = WaVoCallBack()
        # my_callback = WaVoCallBack(chosen_metrics={'nse': mt.nse_series})
        callbacks = [early_stop_callback, my_callback]
        logger = TensorBoardLogger(config['log_dir'], default_hp_metric=False)

        trainer = pl.Trainer(default_root_dir=config['log_dir'],
                             logger=logger,
                             # accelerator="auto",
                             accelerator="gpu",
                             devices=1,
                             # fast_dev_run=True,
                             callbacks=callbacks,
                             max_epochs=50,
                             log_every_n_steps=1,
                             )

        if found_lr is None and tune_lr:
            tuner = pl.tuner.Tuner(trainer)
            # tuner.scale_batch_size(model, datamodule=data_module,init_val=512, max_trials=3)
            lr_finder = tuner.lr_find(model, datamodule=data_module)
            found_lr = lr_finder.suggestion()

        trainer.fit(model, data_module)


def objective(trial):
    # We optimize the number of layers, hidden units in each layer and dropouts.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    output_dims = [
        trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True) for i in range(n_layers)
    ]

    model = LightningNet(dropout, output_dims)
    datamodule = FashionMNISTDataModule(data_dir=DIR, batch_size=BATCHSIZE)

    trainer = pl.Trainer(
        logger=True,
        limit_val_batches=PERCENT_VALID_EXAMPLES,
        enable_checkpointing=False,
        max_epochs=EPOCHS,
        accelerator="auto",
        devices=1,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")],
    )
    hyperparameters = dict(
        n_layers=n_layers, dropout=dropout, output_dims=output_dims)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics["val_acc"].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WaVo Optuna Hyperparameter search.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    args = parser.parse_args()

    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    # TODO
    optuna.pruners.HyperbandPruner(
        min_resource=1, max_resource='auto', reduction_factor=3, bootstrap_count=0)

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
