"""
Main Module to start training
"""
import argparse
import logging
import sys
from pathlib import Path

import lightning.pytorch as pl
import optuna
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from lightning.pytorch.loggers import TensorBoardLogger
import utility as ut
from callbacks import WaVoCallback, OptunaPruningCallback
from data_tools import WaVoDataModule
from models import WaVoLightningModule

use_cuda = torch.cuda.is_available()

if use_cuda:
    accelerator = 'gpu'
    torch.set_float32_matmul_precision('high')
else:
    accelerator = 'cpu'

# CONSTANTS

storage_base = 'sqlite:///../../models_torch/optuna/'

if ut.debugger_is_active():
    default_storage_name = 'sqlite:///../models_torch/optuna/debug_01.db'
    n_trials = 2
    max_epochs = 5
else:
    default_storage_name = 'sqlite:///../../models_torch/optuna/optimization_01.db'
    n_trials = 100
    max_epochs = 50


if use_cuda:
    default_batch_size = 2048
    max_free = 0
    best_device = None
    for i in range(torch.cuda.device_count()):
        free , _ = torch.cuda.mem_get_info()
        if free > max_free:
            max_free = free
            best_device = i
    devices = [best_device]

else:
    devices = 'auto'
    default_batch_size = 256



class Objective:
    def __init__(self, filename, gauge, logdir, in_size=144, out_size=48, batch_size=2048, percentile=0.95, train=0.7, val=.15, test=.15, **kwargs):
        # Hold these implementation specific arguments as the fields of the class.
        self.filename = filename
        self.level_name_org = gauge
        self.log_dir = logdir
        self.in_size = in_size
        self.out_size = out_size
        self.batch_size = batch_size
        self.percentile = percentile
        self.train = train
        self.val = val
        self.test = test

    def __call__(self, trial):

        #TODO gradient clipping?
        monitor = 'hp/val_loss'
        differencing = trial.suggest_int("differencing", 0, 0)
        model_architecture = trial.suggest_categorical(
            "model_architecture", ["classic_lstm"])
        hidden_size_lstm = trial.suggest_int("hidden_size_lstm", 32, 512)
        hidden_size = trial.suggest_int("hidden_size", 32, 512)
        num_layers = trial.suggest_int("n_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        lr = trial.suggest_float("lr", 0.00001, 0.01, log=True)
        # output_dims = [
        #    trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True) for i in range(n_layers)
        # ]

        #TODO to seed or not to seed?
        #pl.seed_everything(42, workers=True)

        data_module = WaVoDataModule(
            str(self.filename),
            self.level_name_org,
            in_size=self.in_size,
            out_size=self.out_size,
            batch_size=self.batch_size,
            differencing=differencing,
            percentile=self.percentile,
            train=self.train,
            val=self.val,
            test=self.test
        )

        data_module.prepare_data()

        model = WaVoLightningModule(
            target_min=data_module.target_min,
            target_max=data_module.target_max,
            feature_count=data_module.feature_count,
            in_size=data_module.hparams.in_size,
            out_size=data_module.hparams.out_size,
            hidden_size_lstm=hidden_size_lstm,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=lr)

        # Callbacks & Logging
        #TODO consider not only saving the weights?
        checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor=monitor,save_weights_only=True)
        pruning_callback = OptunaPruningCallback(trial, monitor=monitor)
        early_stop_callback = EarlyStopping(
            monitor=monitor, mode="min", patience=3)
        # TODO: update when optuna support the new namespace  # pruning_callback = PyTorchLightningPruningCallback(trial, monitor="hp/val_loss")
        my_callback = WaVoCallback()

        callbacks = [checkpoint_callback,pruning_callback, early_stop_callback, my_callback]

        logger = TensorBoardLogger(self.log_dir, default_hp_metric=False)


        trainer = pl.Trainer(default_root_dir=self.log_dir,
                             logger=logger,
                             accelerator=accelerator,
                             devices=devices,
                             callbacks=callbacks,
                             max_epochs=max_epochs,
                             log_every_n_steps=1,
                             )

        trainer.fit(model, data_module)

        trial.set_user_attr("model_path", str(Path(trainer.log_dir).resolve()))

        for metric in ['hp/val_nse', 'hp/val_mae', 'hp/val_mae_flood']:
            for i in [23, 47]:
                trial.set_user_attr(
                    f'{metric}_{i}', my_callback.metrics[metric][i].item())

        return my_callback.metrics[monitor].item()


def parse_args() -> argparse.Namespace:
    """ Parse all the arguments and provides some help in the command line
    """

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description='Execute Hyperparameter optimization with optuna and torchlightning.')
    parser.add_argument('filename', metavar='datafile', type=Path,
                        help='The path to your input data.')
    parser.add_argument('gauge', metavar='gaugename',
                        type=str, help='The name of the gauge column.')
    parser.add_argument('logdir', type=Path,
                        help='set a directory for logs and model checkpoints.')
    parser.add_argument('trials', metavar='trials',type=int, default=n_trials,help='How many trials to run.')

    parser.add_argument('--expname', metavar='experiment_name',type=str, default='nameless',help='The name of the experiment.')
    parser.add_argument('--storagename', metavar='storage_name',type=str, default=None,help='The database for the experiment.')


    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )

    return parser.parse_args()

def main():
    parsed_args = parse_args()
    if parsed_args.pruning:
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource='auto', reduction_factor=3, bootstrap_count=0)
    else:
        pruner = optuna.pruners.NopPruner()

    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(
        logging.StreamHandler(sys.stdout))

    study_name = f"{parsed_args.filename.stem} {parsed_args.expname}" # Unique identifier of the study.

    storage_name = default_storage_name if parsed_args.storagename is None else f"{storage_base}{parsed_args.storagename}.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        pruner=pruner,
        load_if_exists=True)

    study.set_metric_names(["hp/val_loss"])
    objective = Objective(**vars(parsed_args),batch_size=default_batch_size, gc_after_trial=True)
    study.optimize(objective, n_trials=parsed_args.trials, timeout=None)

if __name__ == "__main__":
    main()
