from __future__ import annotations

import numpy as np

from ..workflows.tabular import evaluate_downstream_models
from . import METRIC_REGISTRY
from .base import MatrixMetric


def _require_tabular_kwargs(kwargs: dict[str, object], n_rows: int) -> tuple[int, np.ndarray, np.ndarray, str]:
    if "target_col" not in kwargs:
        raise ValueError("target_col is required for downstream tabular metrics.")
    if "train_row_mask" not in kwargs or "test_row_mask" not in kwargs:
        raise ValueError("train_row_mask and test_row_mask are required for downstream tabular metrics.")
    target_col = int(kwargs["target_col"])
    train_row_mask = np.asarray(kwargs["train_row_mask"], dtype=bool)
    test_row_mask = np.asarray(kwargs["test_row_mask"], dtype=bool)
    if train_row_mask.shape[0] != n_rows or test_row_mask.shape[0] != n_rows:
        raise ValueError("train_row_mask/test_row_mask length mismatch with matrix rows.")
    task = str(kwargs.get("task", "classification"))
    return target_col, train_row_mask, test_row_mask, task


@METRIC_REGISTRY.register("downstream_accuracy_linear")
class DownstreamAccuracyLinear(MatrixMetric):
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, eval_mask: np.ndarray, **kwargs: object) -> float:
        target_col, train_mask, test_mask, task = _require_tabular_kwargs(kwargs, y_true.shape[0])
        scores = evaluate_downstream_models(
            y_true=y_true,
            y_pred=y_pred,
            target_col=target_col,
            train_row_mask=train_mask,
            test_row_mask=test_mask,
            task=task,
        )
        key = "downstream_accuracy_linear" if task == "classification" else "downstream_r2_linear"
        return float(scores[key])


@METRIC_REGISTRY.register("downstream_accuracy_random_forest")
class DownstreamAccuracyRandomForest(MatrixMetric):
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, eval_mask: np.ndarray, **kwargs: object) -> float:
        target_col, train_mask, test_mask, task = _require_tabular_kwargs(kwargs, y_true.shape[0])
        scores = evaluate_downstream_models(
            y_true=y_true,
            y_pred=y_pred,
            target_col=target_col,
            train_row_mask=train_mask,
            test_row_mask=test_mask,
            task=task,
        )
        key = "downstream_accuracy_random_forest" if task == "classification" else "downstream_r2_random_forest"
        return float(scores[key])


@METRIC_REGISTRY.register("downstream_accuracy_xgboost")
class DownstreamAccuracyXGBoost(MatrixMetric):
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, eval_mask: np.ndarray, **kwargs: object) -> float:
        target_col, train_mask, test_mask, task = _require_tabular_kwargs(kwargs, y_true.shape[0])
        scores = evaluate_downstream_models(
            y_true=y_true,
            y_pred=y_pred,
            target_col=target_col,
            train_row_mask=train_mask,
            test_row_mask=test_mask,
            task=task,
        )
        key = "downstream_accuracy_xgboost" if task == "classification" else "downstream_r2_xgboost"
        if key not in scores:
            raise ValueError("xgboost metric requested but xgboost is not installed.")
        return float(scores[key])

