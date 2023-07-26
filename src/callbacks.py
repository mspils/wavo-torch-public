"""
Contains callbacks for LightningModules
"""

from functools import reduce

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback
from optuna.integration import PyTorchLightningPruningCallback
from torch import Tensor
from torch.utils.data import DataLoader

import metrics as mt
import utility as ut


class WaVoCallback(Callback):
    """Callback with 2 tasks
    First initialize all metrics and losses so that they are displayed in the tensorboard hyperparameter tab.
    After fitting the model metrics are calculated for each forecast horizon on the predict, val and test dataloaders
    provided by the used trainers DataModule.
    Each metric is also calculated for the subset of samples that are defined as flood samples.

    SHOULD ALWAYS BE THE LAST CALLBACK IN THE CALLBACK LIST. Uses predict, which overwrites some values in the trainer.

    Args:
        chosen_metrics (dict, optional): Dictionary of metric names and functions. Defaults to {'nse': mt.nse_series}.
    """

    # pylint: disable-next=dangerous-default-value
    def __init__(self, chosen_metrics=['mse', 'mae', 'kge', 'rmse', 'r2', 'nse', 'p10', 'p20']):

        super().__init__()
        self.chosen_metrics = mt.get_metric_dict(chosen_metrics)

    def on_train_start(self, trainer, pl_module):

        # initialize metrics to log in the hyperparameter tab (necessary for the val_loss, others would still work without this)
        # TODO update syntax when 3.9
        metric_placeholders = {s: ut.get_objective_metric(
            s) for s in self.chosen_metrics}
        metric_placeholders = {**metric_placeholders, **
                               {f'{k}_flood': v for k, v in metric_placeholders.items()}}
        metric_placeholders = [{f'hp/{cur_set}_{k}': v for k, v in metric_placeholders.items(
        )} for cur_set in ['train', 'val', 'test']] + [{'hp/val_loss': 0, 'hp/train_loss': 0}]
        metric_placeholders = reduce(
            lambda a, b: {**a, **b}, metric_placeholders)
        pl_module.logger.log_hyperparams(
            pl_module.hparams, metric_placeholders)

    def on_fit_end(self, trainer, pl_module):
        trainer.my_callback_metrics = trainer.callback_metrics.copy()

        self.threshold = trainer.datamodule.threshold
        metric_placeholder = trainer.callback_metrics.copy()
        self.metrics = {}

        #TODO currently metrics are calculated on the cpu.
        y_true, y_pred, y_true_flood, y_pred_flood = self.get_eval_tensors(
            trainer, pl_module, trainer.datamodule.predict_dataloader(), trainer.datamodule.y_train)
        for m_name, m_func in self.chosen_metrics.items():
            self.metrics[f'hp/train_{m_name}'] = torch.nan_to_num(
                m_func(y_true, y_pred))
            self.metrics[f'hp/train_{m_name}_flood'] = torch.nan_to_num(m_func(
                y_true_flood, y_pred_flood))

        y_true, y_pred, y_true_flood, y_pred_flood = self.get_eval_tensors(
            trainer, pl_module, trainer.datamodule.val_dataloader(), trainer.datamodule.y_val)
        for m_name, m_func in self.chosen_metrics.items():
            self.metrics[f'hp/val_{m_name}'] = torch.nan_to_num(
                m_func(y_true, y_pred))
            self.metrics[f'hp/val_{m_name}_flood'] = torch.nan_to_num(m_func(
                y_true_flood, y_pred_flood))

        y_true, y_pred, y_true_flood, y_pred_flood = self.get_eval_tensors(
            trainer, pl_module, trainer.datamodule.test_dataloader(), trainer.datamodule.y_test)
        for m_name, m_func in self.chosen_metrics.items():
            self.metrics[f'hp/test_{m_name}'] = torch.nan_to_num(
                m_func(y_true, y_pred))
            self.metrics[f'hp/test_{m_name}_flood'] = torch.nan_to_num(m_func(
                y_true_flood, y_pred_flood))

        for i in range(pl_module.out_size):
            # log hyperparameters once
            # if i == 0:
            #    pl_module.logger.log_hyperparams(
            #        pl_module.hparams, {k: v[i].item() for k, v in metrics.items()})
            # else:
            pl_module.logger.log_metrics({k: v[i].item()
                                          for k, v in self.metrics.items()}, step=i+1)

        for k, v in metric_placeholder.items():
            self.metrics[k] = v

    def get_eval_tensors(self, trainer: pl.Trainer, pl_module: pl.LightningModule, data_loader: DataLoader, y_true: Tensor):
        """Makes a prediction for the dataloader and inverts scaling for y_pred and y_true.
            Also creates a masked version of y_true and y_pred with flood samples

        Args:
            trainer (pl.Trainer):
            pl_module (pl.LightningModule):
            data_loader (DataLoader):
            y_true (Tensor): True, scaled values

        Returns:
            Tuple: y_true, y_pred, y_true_flood, y_pred_flood
        """
        y_true = y_true * (pl_module.target_max-pl_module.target_min) + pl_module.target_min

        y_pred = trainer.predict(pl_module, data_loader)
        y_pred = torch.concat(y_pred)

        mask = torch.any(torch.greater(y_true, self.threshold), axis=1)

        y_true_flood = y_true[mask]
        y_pred_flood = y_pred[mask]

        return y_true, y_pred, y_true_flood, y_pred_flood


class OptunaPruningCallback(PyTorchLightningPruningCallback, Callback):
    """Custom optuna Pruning Callback, because CUDA/Lightning do not play well with the default one.
    Must be before the WaVoCallBack, because that one clears the callback_metrics. Maybe by calling predict?

    Args:
        PyTorchLightningPruningCallback (_type_): _description_
        pl (_type_): _description_
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
