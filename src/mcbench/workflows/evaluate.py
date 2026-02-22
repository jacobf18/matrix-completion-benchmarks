from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..io import load_bundle, load_matrix, save_json, validate_bundle
from ..metrics import METRIC_REGISTRY


def evaluate_prediction(
    dataset_dir: Path,
    prediction_path: Path,
    metric_names: list[str],
    output_path: Path,
    metric_params: dict[str, dict[str, Any]] | None = None,
) -> None:
    bundle = load_bundle(dataset_dir)
    validate_bundle(bundle)

    y_true = load_matrix(bundle.ground_truth_path)
    eval_mask = np.load(bundle.eval_mask_path).astype(bool)
    y_pred = load_matrix(prediction_path)

    if y_true.shape != eval_mask.shape or y_true.shape != y_pred.shape:
        raise ValueError("Dataset matrices and prediction must have identical shapes.")

    metrics_payload: dict[str, float] = {}
    metric_params = metric_params or {}
    for name in metric_names:
        metric_cls = METRIC_REGISTRY.get(name)
        if metric_cls is None:
            known = ", ".join(sorted(METRIC_REGISTRY))
            raise ValueError(f"Unknown metric '{name}'. Known: {known}")
        metric = metric_cls()
        score = metric.compute(
            y_true=y_true,
            y_pred=y_pred,
            eval_mask=eval_mask,
            **metric_params.get(name, {}),
        )
        metrics_payload[name] = score

    save_json(
        output_path,
        {
            "dataset_dir": str(dataset_dir),
            "prediction_path": str(prediction_path),
            "evaluated_cells": int(np.sum(eval_mask)),
            "metrics": metrics_payload,
        },
    )

