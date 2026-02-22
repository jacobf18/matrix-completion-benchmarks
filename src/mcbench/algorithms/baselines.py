from __future__ import annotations

import numpy as np

from . import ALGORITHM_REGISTRY
from .base import MatrixCompletionAlgorithm


@ALGORITHM_REGISTRY.register("global_mean")
class GlobalMeanImputer(MatrixCompletionAlgorithm):
    def complete(self, observed: np.ndarray, **kwargs: object) -> np.ndarray:
        out = observed.astype(np.float64, copy=True)
        finite = np.isfinite(out)
        if not np.any(finite):
            raise ValueError("Input has no finite entries to estimate a fill value.")
        global_mean = float(np.nanmean(out))
        out[~finite] = global_mean
        return out


@ALGORITHM_REGISTRY.register("row_mean")
class RowMeanImputer(MatrixCompletionAlgorithm):
    def complete(self, observed: np.ndarray, **kwargs: object) -> np.ndarray:
        out = observed.astype(np.float64, copy=True)
        finite = np.isfinite(out)
        if not np.any(finite):
            raise ValueError("Input has no finite entries to estimate fill values.")
        global_mean = float(np.nanmean(out))
        row_means = np.nanmean(out, axis=1)
        row_means = np.where(np.isfinite(row_means), row_means, global_mean)
        missing_rows, missing_cols = np.where(~finite)
        out[missing_rows, missing_cols] = row_means[missing_rows]
        return out


@ALGORITHM_REGISTRY.register("soft_impute")
class SoftImpute(MatrixCompletionAlgorithm):
    def complete(self, observed: np.ndarray, **kwargs: object) -> np.ndarray:
        shrinkage = float(kwargs.get("shrinkage", 1.0))
        max_iters = int(kwargs.get("max_iters", 100))
        tol = float(kwargs.get("tol", 1e-5))
        rank = kwargs.get("rank")
        init_fill = str(kwargs.get("init_fill", "global_mean"))

        if shrinkage < 0:
            raise ValueError("shrinkage must be >= 0.")
        if max_iters < 1:
            raise ValueError("max_iters must be >= 1.")
        if tol <= 0:
            raise ValueError("tol must be > 0.")
        if rank is not None:
            rank = int(rank)
            if rank < 1:
                raise ValueError("rank must be >= 1 when provided.")

        matrix = observed.astype(np.float64, copy=True)
        known = np.isfinite(matrix)
        missing = ~known
        if not np.any(known):
            raise ValueError("Input has no finite entries.")

        if init_fill == "zero":
            matrix[missing] = 0.0
        elif init_fill == "global_mean":
            matrix[missing] = float(np.nanmean(matrix))
        else:
            raise ValueError("init_fill must be one of: 'global_mean', 'zero'.")

        if not np.any(missing):
            return matrix

        for _ in range(max_iters):
            u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
            if rank is not None:
                u = u[:, :rank]
                singular_values = singular_values[:rank]
                vt = vt[:rank, :]

            singular_values = np.maximum(singular_values - shrinkage, 0.0)
            active = singular_values > 0

            if np.any(active):
                low_rank = (u[:, active] * singular_values[active]) @ vt[active, :]
            else:
                low_rank = np.zeros_like(matrix)

            next_matrix = low_rank
            next_matrix[known] = observed[known]

            denom = np.linalg.norm(matrix[missing]) + 1e-12
            delta = np.linalg.norm((next_matrix - matrix)[missing]) / denom
            matrix = next_matrix
            if delta < tol:
                break

        return matrix
