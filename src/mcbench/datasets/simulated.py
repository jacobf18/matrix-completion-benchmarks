from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..io import save_json, save_mask, save_matrix


def generate_simulated_noisy_benchmark(
    output_dir: Path,
    dgp_type: str,
    noise_type: str,
    params: dict[str, Any],
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    n_rows = int(params.get("n_rows", 300))
    n_cols = int(params.get("n_cols", 250))
    rank = int(params.get("rank", 8))
    observed_fraction = float(params.get("observed_fraction", 0.3))
    eval_fraction = float(params.get("eval_fraction", 0.2))

    if n_rows < 2 or n_cols < 2:
        raise ValueError("n_rows and n_cols must be >= 2.")
    if rank < 1 or rank > min(n_rows, n_cols):
        raise ValueError("rank must be in [1, min(n_rows, n_cols)].")
    if not (0 < observed_fraction <= 1):
        raise ValueError("observed_fraction must be in (0, 1].")
    if not (0 < eval_fraction < 1):
        raise ValueError("eval_fraction must be in (0, 1).")

    ground_truth = _generate_ground_truth(dgp_type=dgp_type, n_rows=n_rows, n_cols=n_cols, rank=rank, rng=rng, params=params)
    train_mask, eval_mask = _make_masks(
        shape=ground_truth.shape,
        observed_fraction=observed_fraction,
        eval_fraction=eval_fraction,
        rng=rng,
    )

    noisy_full = _add_noise(matrix=ground_truth, noise_type=noise_type, rng=rng, params=params)

    observed = np.full_like(ground_truth, np.nan)
    observed[train_mask] = noisy_full[train_mask]

    output_dir.mkdir(parents=True, exist_ok=True)
    save_matrix(output_dir / "ground_truth.npy", ground_truth)
    save_matrix(output_dir / "observed.npy", observed)
    save_mask(output_dir / "train_mask.npy", train_mask)
    save_mask(output_dir / "eval_mask.npy", eval_mask)
    save_json(
        output_dir / "simulation_meta.json",
        {
            "dgp_type": dgp_type,
            "noise_type": noise_type,
            "params": params,
            "seed": seed,
            "shape": [n_rows, n_cols],
            "train_count": int(np.sum(train_mask)),
            "eval_count": int(np.sum(eval_mask)),
        },
    )


def _generate_ground_truth(
    dgp_type: str,
    n_rows: int,
    n_cols: int,
    rank: int,
    rng: np.random.Generator,
    params: dict[str, Any],
) -> np.ndarray:
    if dgp_type == "low_rank_gaussian":
        scale = float(params.get("factor_scale", 1.0))
        u = rng.normal(0.0, scale, size=(n_rows, rank))
        v = rng.normal(0.0, scale, size=(n_cols, rank))
        return (u @ v.T).astype(np.float64)

    if dgp_type == "low_rank_orthogonal":
        singular_decay = float(params.get("singular_decay", 0.8))
        u, _ = np.linalg.qr(rng.normal(size=(n_rows, rank)))
        v, _ = np.linalg.qr(rng.normal(size=(n_cols, rank)))
        s = np.array([singular_decay**k for k in range(rank)], dtype=np.float64)
        return (u * s) @ v.T

    if dgp_type == "block_low_rank":
        n_row_groups = int(params.get("n_row_groups", 6))
        n_col_groups = int(params.get("n_col_groups", 5))
        group_noise = float(params.get("group_noise", 0.1))
        row_group = rng.integers(0, n_row_groups, size=n_rows)
        col_group = rng.integers(0, n_col_groups, size=n_cols)
        block = rng.normal(size=(n_row_groups, n_col_groups))
        mat = block[row_group][:, col_group]
        mat += rng.normal(scale=group_noise, size=mat.shape)
        return mat.astype(np.float64)

    raise ValueError(f"Unsupported dgp_type: {dgp_type}")


def _make_masks(
    shape: tuple[int, int],
    observed_fraction: float,
    eval_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    total = shape[0] * shape[1]
    observed_count = max(2, int(round(total * observed_fraction)))
    all_idx = np.arange(total)
    observed_idx = rng.choice(all_idx, size=observed_count, replace=False)
    eval_count = max(1, int(round(observed_count * eval_fraction)))
    eval_idx = rng.choice(observed_idx, size=eval_count, replace=False)
    train_mask = np.zeros(total, dtype=bool)
    eval_mask = np.zeros(total, dtype=bool)
    train_mask[observed_idx] = True
    train_mask[eval_idx] = False
    eval_mask[eval_idx] = True
    return train_mask.reshape(shape), eval_mask.reshape(shape)


def _add_noise(
    matrix: np.ndarray,
    noise_type: str,
    rng: np.random.Generator,
    params: dict[str, Any],
) -> np.ndarray:
    out = matrix.astype(np.float64, copy=True)
    if noise_type == "gaussian":
        sigma = float(params.get("sigma", 0.1))
        out += rng.normal(0.0, sigma, size=out.shape)
        return out

    if noise_type == "student_t":
        df = float(params.get("df", 3.0))
        scale = float(params.get("scale", 0.15))
        out += rng.standard_t(df=df, size=out.shape) * scale
        return out

    if noise_type == "sparse_corruption":
        frac = float(params.get("corruption_fraction", 0.05))
        scale = float(params.get("corruption_scale", 3.0))
        if not (0 <= frac <= 1):
            raise ValueError("corruption_fraction must be in [0, 1].")
        corrupt_count = int(round(out.size * frac))
        if corrupt_count > 0:
            idx = rng.choice(np.arange(out.size), size=corrupt_count, replace=False)
            out.flat[idx] += rng.normal(0.0, scale, size=corrupt_count)
        return out

    raise ValueError(f"Unsupported noise_type: {noise_type}")

