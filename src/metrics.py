"""
Contains all metric functions and maybe some day classes.
"""

import torch
from torch import Tensor


def nse_series(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Calculates NSE for all prediction steps,
    e.g. 48 values if you predict the next 48 hours for each sample
    Args:
        y_true (Tensor): true values
        y_pred (Tensor): predicted values

    Returns:
        Tensor: metric for all forecast horizons.
    """
    nom = torch.sum(torch.square(torch.sub(y_true, y_pred)), axis=0)
    denom = torch.sum(torch.square(
        torch.sub(y_true, torch.mean(y_true, axis=0))), axis=0)

    return 1 - (nom / denom)


def r_squared_series(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Calculates R^2 for all prediction steps,
    e.g. 48 values if you predict the next 48 hours for each sample
    Args:
        y_true (Tensor): true values
        y_pred (Tensor): predicted values

    Returns:
        Tensor: metric for all forecast horizons.
    """
    residual = torch.sum(torch.square(torch.sub(y_true, y_pred)), axis=0)
    total = torch.sum(torch.square(
        torch.sub(y_true, torch.mean(y_true, axis=0))), axis=0)
    return torch.subtract(1, torch.divide(residual, total))


def mse_series(
        y_true: Tensor,
        y_pred: Tensor) -> Tensor:
    """
    Calculates the name giving metric for all prediction steps,
    e.g. 48 values if you predict the next 48 hours for each sample
    Args:
        y_true (Tensor): true values
        y_pred (Tensor): predicted values

    Returns:
        Tensor: namegiving values for all prediction steps
    """
    return torch.mean(torch.square(torch.subtract(y_true, y_pred)), axis=0)


def rmse_series(
        y_true: Tensor,
        y_pred: Tensor) -> Tensor:
    """
    Calculates the name giving metric for all prediction steps,
    e.g. 48 values if you predict the next 48 hours for each sample
    Args:
        y_true (Tensor): true values
        y_pred (Tensor): predicted values

    Returns:
        Tensor: namegiving values for all prediction steps
    """
    return torch.sqrt(torch.mean(torch.square(torch.subtract(y_true, y_pred)), axis=0))


def kge_series(y_true:  Tensor, y_pred:  Tensor) -> Tensor:
    """
    Calculates the name giving metric for all prediction steps,
    e.g. 48 values if you predict the next 48 hours for each sample
    Args:
        y_true (Tensor): true values
        y_pred (Tensor): predicted values

    Returns:
        Tensor: namegiving values for all prediction steps
    """
    # calculate error in timing and dynamics r
    # (Pearson's correlation coefficient)
    sim_mean = torch.mean(y_pred, axis=0)
    obs_mean = torch.mean(y_true, axis=0)

    r_nom = torch.sum((y_pred - sim_mean) * (y_true - obs_mean), axis=0)
    r_denom = torch.sqrt(torch.sum((y_pred - sim_mean) ** 2, axis=0)
                         * torch.sum((y_true - obs_mean) ** 2, axis=0))
    r = r_nom / r_denom

    # calculate error in spread of flow alpha
    alpha = torch.std(y_pred, axis=0) / torch.std(y_true, axis=0)
    # calculate error in volume beta (bias of mean discharge)
    beta = (torch.sum(y_pred, axis=0) / torch.sum(y_true, axis=0))
    # calculate the Kling-Gupta Efficiency KGE
    kge_ = 1 - torch.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    # np.vstack((kge_, r, alpha, beta))
    # return np.vstack((kge_, r, alpha, beta))
    return kge_


def mae_series(y_true, y_pred):
    """
    Calculates the name giving metric for all prediction steps,
    e.g. 48 values if you predict the next 48 hours for each sample
    Args:
        y_true (Tensor): true values
        y_pred (Tensor): predicted values

    Returns:
        Tensor: namegiving values for all prediction steps
    """
    return torch.mean(torch.abs(torch.subtract(y_true, y_pred)), axis=0)


# TODO higher order/currying?
def p10_series(y_true, y_pred):
    """
    Calculates the name giving metric for all prediction steps,
    e.g. 48 values if you predict the next 48 hours for each sample
    Args:
        y_true (Tensor): true values
        y_pred (Tensor): predicted values

    Returns:
        (Tensor,Tensor): namegiving values for all prediction steps
    """
    diff = torch.abs(torch.subtract(y_true, y_pred))
    nom = torch.sum(torch.greater(diff, 10), axis=0)
    return nom / (y_true.shape[0]/100)


def p20_series(y_true, y_pred):
    """
    Calculates the name giving metric for all prediction steps,
    e.g. 48 values if you predict the next 48 hours for each sample
    Args:
        y_true (Tensor): true values
        y_pred (Tensor): predicted values

    Returns:
        (Tensor,Tensor): namegiving values for all prediction steps
    """
    diff = torch.abs(torch.subtract(y_true, y_pred))
    nom = torch.sum(torch.greater(diff, 20), axis=0)
    return nom / (y_true.shape[0]/100)
