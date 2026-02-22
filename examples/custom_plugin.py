from __future__ import annotations

import numpy as np

from mcbench.algorithms import ALGORITHM_REGISTRY
from mcbench.algorithms.base import MatrixCompletionAlgorithm
from mcbench.metrics import METRIC_REGISTRY
from mcbench.metrics.base import MatrixMetric


@ALGORITHM_REGISTRY.register("column_mean")
class ColumnMeanImputer(MatrixCompletionAlgorithm):
    def complete(self, observed: np.ndarray, **kwargs: object) -> np.ndarray:
        out = observed.astype(np.float64, copy=True)
        finite = np.isfinite(out)
        global_mean = float(np.nanmean(out))
        col_means = np.nanmean(out, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, global_mean)
        missing_rows, missing_cols = np.where(~finite)
        out[missing_rows, missing_cols] = col_means[missing_cols]
        return out


@METRIC_REGISTRY.register("max_abs_error")
class MaxAbsoluteError(MatrixMetric):
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, eval_mask: np.ndarray, **kwargs: object) -> float:
        valid = eval_mask & np.isfinite(y_true) & np.isfinite(y_pred)
        return float(np.max(np.abs(y_pred[valid] - y_true[valid])))

