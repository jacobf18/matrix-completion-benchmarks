from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..io import load_matrix, save_json, save_matrix


def make_noisy_matrix(
    input_matrix_path: Path,
    output_matrix_path: Path,
    noise_type: str,
    params: dict[str, Any],
    seed: int = 0,
) -> None:
    matrix = load_matrix(input_matrix_path)
    out = matrix.astype(np.float64, copy=True)
    known = np.isfinite(out)
    rng = np.random.default_rng(seed)

    if noise_type == "gaussian":
        sigma = float(params.get("sigma", 0.25))
        out[known] += rng.normal(loc=0.0, scale=sigma, size=int(np.sum(known)))
        _clip_known(out, known, params)
    elif noise_type == "sparse_corruption":
        frac = float(params.get("corruption_fraction", 0.1))
        scale = float(params.get("corruption_scale", 2.5))
        if not (0 <= frac <= 1):
            raise ValueError("corruption_fraction must be in [0, 1].")
        known_idx = np.flatnonzero(known)
        n_corrupt = int(round(frac * known_idx.size))
        corrupt_idx = rng.choice(known_idx, size=n_corrupt, replace=False) if n_corrupt > 0 else []
        out.flat[corrupt_idx] += rng.normal(loc=0.0, scale=scale, size=n_corrupt)
        _clip_known(out, known, params)
    elif noise_type == "one_bit_flip":
        threshold = float(params.get("threshold", 3.0))
        flip_prob = float(params.get("flip_probability", 0.15))
        if not (0 <= flip_prob <= 1):
            raise ValueError("flip_probability must be in [0, 1].")
        binary = np.full_like(out, np.nan)
        binary[known] = np.where(out[known] >= threshold, 1.0, -1.0)
        flips = rng.random(size=binary.shape) < flip_prob
        active = flips & known
        binary[active] *= -1.0
        out = binary
    else:
        raise ValueError(f"Unsupported noise_type: {noise_type}")

    save_matrix(output_matrix_path, out)
    save_json(
        output_matrix_path.parent / "noise_meta.json",
        {
            "input_matrix_path": str(input_matrix_path),
            "output_matrix_path": str(output_matrix_path),
            "noise_type": noise_type,
            "params": params,
            "seed": seed,
            "known_count": int(np.sum(known)),
        },
    )


def _clip_known(matrix: np.ndarray, known: np.ndarray, params: dict[str, Any]) -> None:
    clip_min = params.get("clip_min")
    clip_max = params.get("clip_max")
    if clip_min is None and clip_max is None:
        return
    lo = -np.inf if clip_min is None else float(clip_min)
    hi = np.inf if clip_max is None else float(clip_max)
    matrix[known] = np.clip(matrix[known], lo, hi)

