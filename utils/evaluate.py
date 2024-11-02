import numpy as np
import pandas as pd

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """
    Calculate the quantile loss between the true and predicted values.

    The quantile loss measures the deviation between the true
        and predicted values at a specific quantile.

    Parameters
    ----------
    y_true : np.ndarray
        The true values.
    y_pred : np.ndarray
        The predicted values.
    quantile : float
        The quantile to calculate the loss for.

    Returns
    -------
    float
        The quantile loss.
    """
    loss = np.mean(
        quantile * np.maximum(y_true - y_pred, 0) + (1 - quantile) * np.maximum(y_pred - y_true, 0)
    )

    return loss
