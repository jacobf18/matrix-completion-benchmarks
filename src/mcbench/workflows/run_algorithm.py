from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ..algorithms import ALGORITHM_REGISTRY
from ..io import load_bundle, load_matrix, save_json, save_matrix, validate_bundle


def run_algorithm(
    dataset_dir: Path,
    algorithm_name: str,
    output_dir: Path,
    algorithm_params: dict[str, Any] | None = None,
) -> None:
    bundle = load_bundle(dataset_dir)
    validate_bundle(bundle)
    observed = load_matrix(bundle.observed_path)

    algo_cls = ALGORITHM_REGISTRY.get(algorithm_name)
    if algo_cls is None:
        known = ", ".join(sorted(ALGORITHM_REGISTRY))
        raise ValueError(f"Unknown algorithm '{algorithm_name}'. Known: {known}")

    algorithm = algo_cls()
    params = algorithm_params or {}

    start = time.perf_counter()
    prediction = algorithm.complete(observed, **params)
    elapsed_s = time.perf_counter() - start

    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != observed.shape:
        raise ValueError(
            f"Algorithm output shape {prediction.shape} does not match observed shape {observed.shape}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "prediction.npy"
    save_matrix(pred_path, prediction)
    save_json(
        output_dir / "run_meta.json",
        {
            "dataset_dir": str(dataset_dir),
            "algorithm_name": algorithm_name,
            "algorithm_params": params,
            "prediction_path": str(pred_path),
            "runtime_seconds": elapsed_s,
        },
    )

