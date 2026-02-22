from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io import load_matrix, save_json, save_mask, save_matrix


def prepare_random_holdout(
    input_matrix_path: Path,
    output_dataset_dir: Path,
    holdout_fraction: float,
    seed: int,
) -> None:
    if not (0 < holdout_fraction < 1):
        raise ValueError("holdout_fraction must be between 0 and 1.")

    matrix = load_matrix(input_matrix_path)
    if matrix.ndim != 2:
        raise ValueError("Input matrix must be 2D.")

    observed = matrix.astype(np.float64, copy=True)
    known_mask = np.isfinite(observed)
    known_count = int(np.sum(known_mask))
    if known_count < 2:
        raise ValueError("Need at least 2 known entries to create a holdout split.")

    n_eval = max(1, int(round(known_count * holdout_fraction)))
    rng = np.random.default_rng(seed)
    known_indices = np.flatnonzero(known_mask)
    eval_indices = rng.choice(known_indices, size=n_eval, replace=False)

    eval_mask = np.zeros_like(known_mask, dtype=bool)
    eval_mask.flat[eval_indices] = True

    observed.flat[eval_indices] = np.nan

    output_dataset_dir.mkdir(parents=True, exist_ok=True)
    save_matrix(output_dataset_dir / "observed.npy", observed)
    save_matrix(output_dataset_dir / "ground_truth.npy", matrix)
    save_mask(output_dataset_dir / "eval_mask.npy", eval_mask)
    save_json(
        output_dataset_dir / "dataset_meta.json",
        {
            "input_matrix_path": str(input_matrix_path),
            "holdout_fraction": holdout_fraction,
            "seed": seed,
            "shape": list(matrix.shape),
            "known_count": known_count,
            "eval_count": int(np.sum(eval_mask)),
        },
    )

