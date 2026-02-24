#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mcbench.io import load_matrix, save_json
from mcbench.workflows.tabular import (
    evaluate_downstream_models,
    evaluate_multiple_imputation_metrics,
    evaluate_single_imputation_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate single or multiple tabular imputations with cellwise and downstream metrics."
    )
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--prediction-path", action="append", required=True, type=Path)
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--target-col", type=int, default=None)
    parser.add_argument("--output-path", required=True, type=Path)
    args = parser.parse_args()

    y_true = load_matrix(args.dataset_dir / "ground_truth.npy")
    eval_mask = np.load(args.dataset_dir / "eval_mask.npy").astype(bool)
    train_mask = np.load(args.dataset_dir / "downstream_train_mask.npy").astype(bool)
    test_mask = np.load(args.dataset_dir / "downstream_test_mask.npy").astype(bool)

    if args.target_col is None:
        meta_path = args.dataset_dir / "dataset_meta.json"
        if not meta_path.exists():
            raise ValueError("target-col not provided and dataset_meta.json missing.")
        meta = json.loads(meta_path.read_text())
        target_col = int(meta["target_col"])
    else:
        target_col = int(args.target_col)

    predictions = [load_matrix(path) for path in args.prediction_path]
    if any(pred.shape != y_true.shape for pred in predictions):
        raise ValueError("All prediction matrices must match ground truth shape.")

    payload: dict[str, object] = {
        "dataset_dir": str(args.dataset_dir),
        "prediction_paths": [str(path) for path in args.prediction_path],
        "task": args.task,
        "target_col": target_col,
        "num_predictions": len(predictions),
    }

    if len(predictions) == 1:
        pred = predictions[0]
        payload["imputation_metrics"] = evaluate_single_imputation_metrics(
            y_true=y_true,
            y_pred=pred,
            eval_mask=eval_mask,
        )
        payload["downstream_metrics"] = evaluate_downstream_models(
            y_true=y_true,
            y_pred=pred,
            target_col=target_col,
            train_row_mask=train_mask,
            test_row_mask=test_mask,
            task=args.task,
        )
    else:
        payload["multiple_imputation_metrics"] = evaluate_multiple_imputation_metrics(
            y_true=y_true,
            y_preds=predictions,
            eval_mask=eval_mask,
            target_col=target_col,
            train_row_mask=train_mask,
            test_row_mask=test_mask,
            task=args.task,
        )

    save_json(args.output_path, payload)
    print(f"wrote evaluation: {args.output_path}")


if __name__ == "__main__":
    main()

