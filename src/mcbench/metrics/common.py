from __future__ import annotations

import numpy as np

from . import METRIC_REGISTRY
from .base import MatrixMetric


def _masked_values(y_true: np.ndarray, y_pred: np.ndarray, eval_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if y_true.shape != y_pred.shape or y_true.shape != eval_mask.shape:
        raise ValueError("Shape mismatch among y_true, y_pred, and eval_mask.")
    valid = eval_mask & np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        raise ValueError("No valid entries to evaluate.")
    return y_true[valid], y_pred[valid]


@METRIC_REGISTRY.register("rmse")
class RMSE(MatrixMetric):
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, eval_mask: np.ndarray, **kwargs: object) -> float:
        t, p = _masked_values(y_true, y_pred, eval_mask)
        return float(np.sqrt(np.mean((p - t) ** 2)))


@METRIC_REGISTRY.register("mae")
class MAE(MatrixMetric):
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, eval_mask: np.ndarray, **kwargs: object) -> float:
        t, p = _masked_values(y_true, y_pred, eval_mask)
        return float(np.mean(np.abs(p - t)))


@METRIC_REGISTRY.register("nmae")
class NormalizedMAE(MatrixMetric):
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, eval_mask: np.ndarray, **kwargs: object) -> float:
        t, p = _masked_values(y_true, y_pred, eval_mask)
        denom = np.max(t) - np.min(t)
        if denom == 0:
            return 0.0
        return float(np.mean(np.abs(p - t)) / denom)

