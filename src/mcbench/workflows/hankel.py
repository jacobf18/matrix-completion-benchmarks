from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..algorithms import ALGORITHM_REGISTRY


@dataclass(frozen=True)
class HankelBenchmarkData:
    clean_signal: np.ndarray
    noisy_signal: np.ndarray
    observed_signal: np.ndarray
    observed_mask: np.ndarray
    missing_mask: np.ndarray
    hankel_observed: np.ndarray


def generate_hankel_benchmark(
    signal_model: str,
    mask_type: str,
    params: dict[str, Any],
    seed: int,
) -> HankelBenchmarkData:
    rng = np.random.default_rng(seed)
    n = int(params.get("n", 256))
    rank = int(params.get("rank", 8))
    if n < 8:
        raise ValueError("n must be >= 8")
    if rank < 1 or rank > max(1, n // 2):
        raise ValueError("rank must be in [1, n//2]")

    if signal_model == "spectrally_sparse_complex":
        clean = _spectrally_sparse_signal(n=n, rank=rank, rng=rng, damping=False)
    elif signal_model == "damped_spectrally_sparse":
        clean = _spectrally_sparse_signal(n=n, rank=rank, rng=rng, damping=True)
    else:
        raise ValueError(f"Unsupported signal_model: {signal_model}")

    noise_sigma = float(params.get("noise_sigma", 0.02))
    noisy = clean + rng.normal(0.0, noise_sigma, size=n)

    missing_fraction = float(params.get("missing_fraction", 0.5))
    if not (0 < missing_fraction < 1):
        raise ValueError("missing_fraction must be in (0,1)")
    observed = np.ones(n, dtype=bool)
    n_missing = int(round(n * missing_fraction))

    if mask_type == "uniform_random":
        miss_idx = rng.choice(np.arange(n), size=n_missing, replace=False)
        observed[miss_idx] = False
    elif mask_type == "contiguous_block":
        start = int(rng.integers(0, max(1, n - n_missing)))
        observed[start : start + n_missing] = False
    elif mask_type == "bursty_random":
        observed[:] = True
        while int(np.sum(~observed)) < n_missing:
            remaining = n_missing - int(np.sum(~observed))
            burst = int(rng.integers(2, min(12, remaining + 2)))
            start = int(rng.integers(0, max(1, n - burst)))
            observed[start : start + burst] = False
        over = int(np.sum(~observed) - n_missing)
        if over > 0:
            false_idx = np.flatnonzero(~observed)
            restore = rng.choice(false_idx, size=over, replace=False)
            observed[restore] = True
    else:
        raise ValueError(f"Unsupported mask_type: {mask_type}")

    observed_signal = noisy.copy()
    observed_signal[~observed] = np.nan

    window = int(params.get("window_length", n // 2))
    hankel_observed = hankelize(observed_signal, window=window)
    return HankelBenchmarkData(
        clean_signal=clean,
        noisy_signal=noisy,
        observed_signal=observed_signal,
        observed_mask=observed,
        missing_mask=~observed,
        hankel_observed=hankel_observed,
    )


def hankelize(signal: np.ndarray, window: int) -> np.ndarray:
    n = signal.shape[0]
    if window < 2 or window >= n:
        raise ValueError("window must be in [2, n-1]")
    k = n - window + 1
    h = np.empty((window, k), dtype=np.float64)
    for i in range(window):
        h[i, :] = signal[i : i + k]
    return h


def dehankelize(hankel_matrix: np.ndarray) -> np.ndarray:
    l, k = hankel_matrix.shape
    n = l + k - 1
    out = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    for i in range(l):
        for j in range(k):
            val = hankel_matrix[i, j]
            if np.isfinite(val):
                idx = i + j
                out[idx] += val
                counts[idx] += 1.0
    valid = counts > 0
    out[valid] /= counts[valid]
    out[~valid] = np.nan
    return out


def enforce_hankel_structure(matrix: np.ndarray) -> np.ndarray:
    l, k = matrix.shape
    n = l + k - 1
    sums = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    for i in range(l):
        for j in range(k):
            val = matrix[i, j]
            if np.isfinite(val):
                idx = i + j
                sums[idx] += val
                counts[idx] += 1.0
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    out = np.empty_like(matrix, dtype=np.float64)
    for i in range(l):
        for j in range(k):
            out[i, j] = means[i + j]
    return out


def cadzow_complete(hankel_observed: np.ndarray, rank: int, max_iters: int = 100, tol: float = 1e-6) -> np.ndarray:
    observed = np.isfinite(hankel_observed)
    if not np.any(observed):
        raise ValueError("No observed entries in Hankel matrix.")
    x = hankel_observed.copy()
    mean_val = float(np.nanmean(x))
    x[~observed] = mean_val

    prev = x
    for _ in range(max_iters):
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        r = min(rank, s.shape[0])
        low_rank = (u[:, :r] * s[:r]) @ vt[:r, :]
        hankelized = enforce_hankel_structure(low_rank)
        hankelized[observed] = hankel_observed[observed]
        diff = np.linalg.norm(hankelized - prev) / (np.linalg.norm(prev) + 1e-12)
        prev = hankelized
        x = hankelized
        if diff < tol:
            break
    return x


def reconstruct_signal_with_method(
    hankel_observed: np.ndarray,
    algorithm_name: str,
    algorithm_params: dict[str, Any] | None = None,
    hankel_rank: int | None = None,
) -> np.ndarray:
    params = algorithm_params or {}
    if algorithm_name == "cadzow":
        if hankel_rank is None:
            raise ValueError("hankel_rank is required for cadzow.")
        completed_hankel = cadzow_complete(
            hankel_observed=hankel_observed,
            rank=hankel_rank,
            max_iters=int(params.get("max_iters", 100)),
            tol=float(params.get("tol", 1e-6)),
        )
    else:
        algo_cls = ALGORITHM_REGISTRY.get(algorithm_name)
        if algo_cls is None:
            known = ", ".join(sorted(ALGORITHM_REGISTRY))
            raise ValueError(f"Unknown algorithm '{algorithm_name}'. Known: {known}, cadzow")
        completed_hankel = np.asarray(algo_cls().complete(hankel_observed, **params), dtype=np.float64)

    structured = enforce_hankel_structure(completed_hankel)
    return dehankelize(structured)


def evaluate_reconstruction(
    clean_signal: np.ndarray,
    reconstructed_signal: np.ndarray,
    missing_mask: np.ndarray,
) -> dict[str, float]:
    if clean_signal.shape != reconstructed_signal.shape or clean_signal.shape != missing_mask.shape:
        raise ValueError("Signal/mask shape mismatch.")
    finite = np.isfinite(reconstructed_signal)
    valid_all = finite
    valid_missing = finite & missing_mask
    if not np.any(valid_all):
        raise ValueError("No finite reconstructed entries.")
    if not np.any(valid_missing):
        raise ValueError("No finite reconstructed missing entries.")

    diff_all = reconstructed_signal[valid_all] - clean_signal[valid_all]
    diff_missing = reconstructed_signal[valid_missing] - clean_signal[valid_missing]
    rmse_all = float(np.sqrt(np.mean(diff_all**2)))
    rmse_missing = float(np.sqrt(np.mean(diff_missing**2)))
    denom = float(np.max(clean_signal[missing_mask]) - np.min(clean_signal[missing_mask]))
    nrmse_missing = 0.0 if denom == 0 else rmse_missing / denom
    return {
        "rmse_all": rmse_all,
        "rmse_missing": rmse_missing,
        "nrmse_missing": nrmse_missing,
        "reconstructed_missing_count": float(np.sum(valid_missing)),
    }


def _spectrally_sparse_signal(n: int, rank: int, rng: np.random.Generator, damping: bool) -> np.ndarray:
    t = np.arange(n, dtype=np.float64)
    x = np.zeros(n, dtype=np.float64)
    for _ in range(rank):
        amp = float(rng.uniform(0.5, 1.5))
        freq = float(rng.uniform(0.0, np.pi))
        phase = float(rng.uniform(0.0, 2 * np.pi))
        if damping:
            decay = float(rng.uniform(0.0, 0.015))
            x += amp * np.exp(-decay * t) * np.cos(freq * t + phase)
        else:
            x += amp * np.cos(freq * t + phase)
    return x / max(np.std(x), 1e-12)
