#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from mcbench.algorithms import ALGORITHM_REGISTRY
from mcbench.io import load_matrix, save_json, save_matrix
from mcbench.workflows.tabular import (
    evaluate_downstream_models,
    evaluate_multiple_imputation_metrics,
    evaluate_single_imputation_metrics,
    generate_multiple_imputations_gaussian,
)


def _parse_csv_list(raw: str, cast):
    vals = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(cast(token))
    if not vals:
        raise ValueError("Expected at least one value.")
    return vals


def _discover_bundles(dataset_root: Path) -> list[Path]:
    bundles = []
    for meta in sorted(dataset_root.glob("pattern_*__frac_*__seed_*/dataset_meta.json")):
        bundles.append(meta.parent)
    if not bundles:
        raise ValueError(f"No benchmark bundles found in {dataset_root}")
    return bundles


def _metric_summary_row(
    bundle_meta: dict[str, Any],
    method: str,
    metrics: dict[str, float],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "bundle": str(bundle_meta.get("bundle_id", "")),
        "pattern": bundle_meta.get("pattern", ""),
        "missing_fraction": bundle_meta.get("missing_fraction", ""),
        "seed": bundle_meta.get("seed", ""),
        "method": method,
    }
    row.update(metrics)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CKD EHR tabular imputation + downstream supervised evaluation across missingness bundles."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--algorithms", default="global_mean,row_mean,soft_impute")
    parser.add_argument("--algorithm-params-json", default="{}")
    parser.add_argument("--num-imputations", type=int, default=5)
    parser.add_argument("--mi-seed", type=int, default=7)
    parser.add_argument("--task", choices=["classification", "regression"], default="regression")
    parser.add_argument(
        "--include-mi-gaussian",
        action="store_true",
        help="Include the mi_gaussian multiple-imputation baseline in outputs.",
    )
    args = parser.parse_args()

    algorithm_names = _parse_csv_list(args.algorithms, str)
    algorithm_params = json.loads(args.algorithm_params_json)
    if not isinstance(algorithm_params, dict):
        raise ValueError("algorithm-params-json must be a JSON object")

    bundles = _discover_bundles(args.dataset_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []

    for bundle_dir in bundles:
        meta = json.loads((bundle_dir / "dataset_meta.json").read_text())
        meta["bundle_id"] = bundle_dir.name

        y_true = load_matrix(bundle_dir / "ground_truth.npy")
        observed = load_matrix(bundle_dir / "observed.npy")
        eval_mask = np.load(bundle_dir / "eval_mask.npy").astype(bool)
        train_mask = np.load(bundle_dir / "downstream_train_mask.npy").astype(bool)
        test_mask = np.load(bundle_dir / "downstream_test_mask.npy").astype(bool)
        target_col = int(meta["target_col"])

        bundle_out = args.output_root / bundle_dir.name
        bundle_out.mkdir(parents=True, exist_ok=True)

        for algo_name in algorithm_names:
            algo_cls = ALGORITHM_REGISTRY.get(algo_name)
            if algo_cls is None:
                summary_rows.append(
                    {
                        "bundle": bundle_dir.name,
                        "pattern": meta.get("pattern", ""),
                        "missing_fraction": meta.get("missing_fraction", ""),
                        "seed": meta.get("seed", ""),
                        "method": algo_name,
                        "status": "failed",
                        "error": f"unknown algorithm '{algo_name}'",
                    }
                )
                continue

            params = algorithm_params.get(algo_name, {})
            if not isinstance(params, dict):
                raise ValueError(f"algorithm params for '{algo_name}' must be an object")

            try:
                pred = np.asarray(algo_cls().complete(observed, **params), dtype=np.float64)
                impute_metrics = evaluate_single_imputation_metrics(y_true=y_true, y_pred=pred, eval_mask=eval_mask)
                down_metrics = evaluate_downstream_models(
                    y_true=y_true,
                    y_pred=pred,
                    target_col=target_col,
                    train_row_mask=train_mask,
                    test_row_mask=test_mask,
                    task=args.task,
                )
                payload = {
                    "dataset_dir": str(bundle_dir),
                    "method": algo_name,
                    "task": args.task,
                    "target_col": target_col,
                    "imputation_metrics": impute_metrics,
                    "downstream_metrics": down_metrics,
                }
                save_matrix(bundle_out / f"{algo_name}_prediction.npy", pred)
                save_json(bundle_out / f"{algo_name}_eval.json", payload)
                summary_rows.append(
                    _metric_summary_row(
                        bundle_meta=meta,
                        method=algo_name,
                        metrics={
                            **{f"imputation_{k}": v for k, v in impute_metrics.items()},
                            **down_metrics,
                            "status": "ok",
                        },
                    )
                )
            except Exception as exc:
                summary_rows.append(
                    {
                        "bundle": bundle_dir.name,
                        "pattern": meta.get("pattern", ""),
                        "missing_fraction": meta.get("missing_fraction", ""),
                        "seed": meta.get("seed", ""),
                        "method": algo_name,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        if args.include_mi_gaussian:
            try:
                mi_preds = generate_multiple_imputations_gaussian(
                    observed=observed,
                    num_imputations=args.num_imputations,
                    seed=args.mi_seed,
                )
                mi_metrics = evaluate_multiple_imputation_metrics(
                    y_true=y_true,
                    y_preds=mi_preds,
                    eval_mask=eval_mask,
                    target_col=target_col,
                    train_row_mask=train_mask,
                    test_row_mask=test_mask,
                    task=args.task,
                )
                for idx, pred in enumerate(mi_preds):
                    save_matrix(bundle_out / f"mi_gaussian_prediction_{idx:03d}.npy", pred)
                save_json(
                    bundle_out / "mi_gaussian_eval.json",
                    {
                        "dataset_dir": str(bundle_dir),
                        "method": "mi_gaussian",
                        "task": args.task,
                        "target_col": target_col,
                        "num_imputations": args.num_imputations,
                        "multiple_imputation_metrics": mi_metrics,
                    },
                )
                summary_rows.append(
                    _metric_summary_row(
                        bundle_meta=meta,
                        method="mi_gaussian",
                        metrics={**mi_metrics, "status": "ok"},
                    )
                )
            except Exception as exc:
                summary_rows.append(
                    {
                        "bundle": bundle_dir.name,
                        "pattern": meta.get("pattern", ""),
                        "missing_fraction": meta.get("missing_fraction", ""),
                        "seed": meta.get("seed", ""),
                        "method": "mi_gaussian",
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    summary_csv = args.output_root / "ckd_ehr_regression_summary.csv"
    fieldnames = sorted({key for row in summary_rows for key in row.keys()})
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"wrote CKD EHR regression summary: {summary_csv}")


if __name__ == "__main__":
    main()
