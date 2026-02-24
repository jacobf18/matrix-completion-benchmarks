from __future__ import annotations

from typing import Literal

import numpy as np


PatternKind = Literal["mcar", "mar_logistic", "mnar_self_logistic", "block", "bursty"]


def generate_missingness_mask(
    matrix: np.ndarray,
    kind: PatternKind,
    missing_fraction: float,
    seed: int = 0,
    *,
    feature_col: int | None = None,
    block_axis: Literal["rows", "cols"] = "rows",
    burst_max: int = 12,
) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D.")
    if not (0 < missing_fraction < 1):
        raise ValueError("missing_fraction must be in (0, 1).")

    rng = np.random.default_rng(seed)
    finite = np.isfinite(matrix)
    n_total = int(np.sum(finite))
    if n_total < 2:
        raise ValueError("Need at least two finite entries to generate missingness.")
    n_missing = max(1, int(round(n_total * missing_fraction)))

    if kind == "mcar":
        return _mcar(finite=finite, n_missing=n_missing, rng=rng)
    if kind == "mar_logistic":
        return _mar_logistic(matrix=matrix, finite=finite, n_missing=n_missing, rng=rng, feature_col=feature_col)
    if kind == "mnar_self_logistic":
        return _mnar_self_logistic(matrix=matrix, finite=finite, n_missing=n_missing, rng=rng)
    if kind == "block":
        return _block_missing(finite=finite, n_missing=n_missing, rng=rng, axis=block_axis)
    if kind == "bursty":
        return _bursty_missing(finite=finite, n_missing=n_missing, rng=rng, burst_max=burst_max)
    raise ValueError(f"Unsupported missingness kind: {kind}")


def apply_missingness(matrix: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
    if matrix.shape != missing_mask.shape:
        raise ValueError("matrix and missing_mask shapes must match.")
    out = matrix.astype(np.float64, copy=True)
    out[missing_mask] = np.nan
    return out


def _mcar(finite: np.ndarray, n_missing: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(finite)
    chosen = rng.choice(idx, size=n_missing, replace=False)
    mask = np.zeros_like(finite, dtype=bool)
    mask.flat[chosen] = True
    return mask


def _mar_logistic(
    matrix: np.ndarray,
    finite: np.ndarray,
    n_missing: int,
    rng: np.random.Generator,
    feature_col: int | None,
) -> np.ndarray:
    n_rows, n_cols = matrix.shape
    col = int(feature_col) if feature_col is not None else 0
    if not (0 <= col < n_cols):
        raise ValueError("feature_col out of bounds.")
    x = matrix[:, col].astype(np.float64, copy=True)
    if not np.any(np.isfinite(x)):
        return _mcar(finite=finite, n_missing=n_missing, rng=rng)

    x_finite = x[np.isfinite(x)]
    mu = float(np.mean(x_finite))
    sd = float(np.std(x_finite))
    z = np.zeros_like(x)
    if sd > 0:
        z[np.isfinite(x)] = (x[np.isfinite(x)] - mu) / sd
    logits = 1.2 * z + rng.normal(0.0, 0.2, size=n_rows)
    row_probs = _sigmoid(logits)
    row_probs = row_probs / np.sum(row_probs)

    row_ids = rng.choice(np.arange(n_rows), size=n_missing * 3, replace=True, p=row_probs)
    miss = np.zeros_like(finite, dtype=bool)
    flat_candidates = []
    for r in row_ids:
        cols = np.flatnonzero(finite[r])
        if cols.size == 0:
            continue
        c = int(rng.choice(cols))
        flat_candidates.append(r * n_cols + c)
    if not flat_candidates:
        return _mcar(finite=finite, n_missing=n_missing, rng=rng)
    unique = np.array(list(dict.fromkeys(flat_candidates)), dtype=np.int64)
    if unique.size < n_missing:
        extra_pool = np.setdiff1d(np.flatnonzero(finite), unique, assume_unique=False)
        add = rng.choice(extra_pool, size=n_missing - unique.size, replace=False)
        unique = np.concatenate([unique, add])
    miss.flat[unique[:n_missing]] = True
    return miss


def _mnar_self_logistic(
    matrix: np.ndarray,
    finite: np.ndarray,
    n_missing: int,
    rng: np.random.Generator,
) -> np.ndarray:
    vals = matrix[finite]
    mu = float(np.mean(vals))
    sd = float(np.std(vals))
    z = np.zeros_like(matrix, dtype=np.float64)
    if sd > 0:
        z[finite] = (matrix[finite] - mu) / sd
    logits = 1.0 * z[finite] + rng.normal(0.0, 0.15, size=vals.shape[0])
    probs = _sigmoid(logits)
    probs = probs / np.sum(probs)
    finite_idx = np.flatnonzero(finite)
    chosen = rng.choice(finite_idx, size=n_missing, replace=False, p=probs)
    miss = np.zeros_like(finite, dtype=bool)
    miss.flat[chosen] = True
    return miss


def _block_missing(
    finite: np.ndarray,
    n_missing: int,
    rng: np.random.Generator,
    axis: Literal["rows", "cols"],
) -> np.ndarray:
    n_rows, n_cols = finite.shape
    miss = np.zeros_like(finite, dtype=bool)
    if axis == "rows":
        block_rows = max(1, int(round(n_missing / max(n_cols, 1))))
        block_rows = min(block_rows, n_rows)
        start = int(rng.integers(0, max(1, n_rows - block_rows + 1)))
        miss[start : start + block_rows, :] = finite[start : start + block_rows, :]
    else:
        block_cols = max(1, int(round(n_missing / max(n_rows, 1))))
        block_cols = min(block_cols, n_cols)
        start = int(rng.integers(0, max(1, n_cols - block_cols + 1)))
        miss[:, start : start + block_cols] = finite[:, start : start + block_cols]

    current = int(np.sum(miss))
    if current > n_missing:
        idx = np.flatnonzero(miss)
        restore = rng.choice(idx, size=current - n_missing, replace=False)
        miss.flat[restore] = False
    elif current < n_missing:
        pool = np.flatnonzero(finite & ~miss)
        add = rng.choice(pool, size=n_missing - current, replace=False)
        miss.flat[add] = True
    return miss


def _bursty_missing(
    finite: np.ndarray,
    n_missing: int,
    rng: np.random.Generator,
    burst_max: int,
) -> np.ndarray:
    n_rows, n_cols = finite.shape
    miss = np.zeros_like(finite, dtype=bool)
    if burst_max < 2:
        burst_max = 2
    while int(np.sum(miss)) < n_missing:
        remaining = n_missing - int(np.sum(miss))
        burst = int(rng.integers(2, min(burst_max, remaining + 2)))
        if bool(rng.integers(0, 2)):
            # Horizontal burst in one row.
            r = int(rng.integers(0, n_rows))
            c0 = int(rng.integers(0, max(1, n_cols - burst + 1)))
            c1 = min(n_cols, c0 + burst)
            miss[r, c0:c1] |= finite[r, c0:c1]
        else:
            # Vertical burst in one column.
            c = int(rng.integers(0, n_cols))
            r0 = int(rng.integers(0, max(1, n_rows - burst + 1)))
            r1 = min(n_rows, r0 + burst)
            miss[r0:r1, c] |= finite[r0:r1, c]

        current = int(np.sum(miss))
        if current > n_missing:
            idx = np.flatnonzero(miss)
            restore = rng.choice(idx, size=current - n_missing, replace=False)
            miss.flat[restore] = False
    return miss


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
