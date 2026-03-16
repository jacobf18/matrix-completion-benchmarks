#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mcbench.algorithms import ALGORITHM_REGISTRY
from mcbench.io import load_matrix, save_json, save_matrix
from mcbench.workflows.tabular import (
    evaluate_downstream_models_with_predictions,
    evaluate_multiple_imputation_metrics,
    evaluate_single_imputation_metrics,
    generate_multiple_imputations_gaussian,
)


def _progress(msg: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(msg, flush=True)


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
    for meta in sorted(dataset_root.rglob("dataset_meta.json")):
        bundles.append(meta.parent)
    if not bundles:
        raise ValueError(f"No benchmark bundles found in {dataset_root}")
    return bundles


def _metric_summary_row(
    bundle_meta: dict[str, Any],
    method: str,
    metrics: dict[str, float],
) -> dict[str, Any]:
    metrics_with_aliases = dict(metrics)
    for key, val in list(metrics.items()):
        if key.startswith("downstream_average_precision_"):
            pr_auc_key = key.replace("downstream_average_precision_", "downstream_pr_auc_", 1)
            metrics_with_aliases[pr_auc_key] = val

    row: dict[str, Any] = {
        "bundle": str(bundle_meta.get("bundle_id", "")),
        "pattern": bundle_meta.get("pattern", ""),
        "missing_fraction": bundle_meta.get("missing_fraction", ""),
        "seed": bundle_meta.get("seed", ""),
        "method": method,
    }
    row.update(metrics_with_aliases)
    return row


def _save_bundle_snapshot(
    bundle_out: Path,
    bundle_dir: Path,
    meta: dict[str, Any],
    y_true: np.ndarray,
    observed: np.ndarray,
    eval_mask: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> Path:
    snapshot_dir = bundle_out / "repro" / "bundle_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    save_matrix(snapshot_dir / "ground_truth.npy", y_true)
    save_matrix(snapshot_dir / "observed.npy", observed)
    np.save(snapshot_dir / "eval_mask.npy", eval_mask.astype(bool))
    np.save(snapshot_dir / "downstream_train_mask.npy", train_mask.astype(bool))
    np.save(snapshot_dir / "downstream_test_mask.npy", test_mask.astype(bool))
    save_json(
        snapshot_dir / "snapshot_meta.json",
        {
            "source_bundle_dir": str(bundle_dir),
            "bundle_meta": meta,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return snapshot_dir


def _save_algorithm_task_artifacts(
    bundle_out: Path,
    algo_name: str,
    y_pred: np.ndarray,
    downstream_payload: dict[str, Any],
) -> None:
    task_dir = bundle_out / "repro" / "tasks" / algo_name
    task_dir.mkdir(parents=True, exist_ok=True)
    save_matrix(task_dir / "prediction_full.npy", y_pred)

    y_test = np.asarray(downstream_payload["y_test"])
    test_idx = np.asarray(downstream_payload["test_row_indices"])
    train_idx = np.asarray(downstream_payload["train_row_indices"])
    np.save(task_dir / "y_test_true.npy", y_test)
    np.save(task_dir / "test_row_indices.npy", test_idx.astype(np.int64))
    np.save(task_dir / "train_row_indices.npy", train_idx.astype(np.int64))

    prediction_files: dict[str, str] = {}
    for model_name, pred in dict(downstream_payload["predictions"]).items():
        if pred is None:
            continue
        pred_path = task_dir / f"{model_name}_test_pred.npy"
        np.save(pred_path, np.asarray(pred))
        prediction_files[f"{model_name}_test_pred"] = str(pred_path)

    score_files: dict[str, str] = {}
    for model_name, pred_score in dict(downstream_payload["score_predictions"]).items():
        if pred_score is None:
            continue
        score_path = task_dir / f"{model_name}_test_score.npy"
        np.save(score_path, np.asarray(pred_score))
        score_files[f"{model_name}_test_score"] = str(score_path)

    save_json(
        task_dir / "task_artifacts_meta.json",
        {
            "algorithm": algo_name,
            "scores": downstream_payload["scores"],
            "prediction_files": prediction_files,
            "score_files": score_files,
            "n_test_rows": int(test_idx.size),
            "n_train_rows": int(train_idx.size),
        },
    )


def _save_algorithm_error_artifact(
    bundle_out: Path,
    algo_name: str,
    error_message: str,
) -> None:
    error_dir = bundle_out / "repro" / "errors"
    error_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        error_dir / f"{algo_name}.json",
        {
            "algorithm": algo_name,
            "status": "failed",
            "error": error_message,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def _complete_with_split_guard(
    algo_cls: Any,
    observed: np.ndarray,
    y_true: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    target_col: int,
    split_mode: str,
    params: dict[str, Any],
) -> np.ndarray:
    if split_mode == "transductive":
        return np.asarray(algo_cls().complete(observed, **params), dtype=np.float64)
    if split_mode != "row_disjoint":
        raise ValueError(f"Unknown imputation split mode: {split_mode}")

    # Train pass: algorithm sees only train rows.
    train_input = np.full_like(observed, np.nan, dtype=np.float64)
    train_input[train_mask, :] = observed[train_mask, :]
    train_pred = np.asarray(algo_cls().complete(train_input, **params), dtype=np.float64)

    # Test pass: algorithm sees only test rows and never sees test targets.
    test_input = np.full_like(observed, np.nan, dtype=np.float64)
    test_input[test_mask, :] = observed[test_mask, :]
    test_input[test_mask, target_col] = np.nan
    test_pred = np.asarray(algo_cls().complete(test_input, **params), dtype=np.float64)

    combined = np.asarray(observed, dtype=np.float64).copy()
    combined[train_mask, :] = train_pred[train_mask, :]
    combined[test_mask, :] = test_pred[test_mask, :]
    # Downstream labels always come from y_true, but keep target column aligned.
    combined[:, target_col] = y_true[:, target_col]
    return combined


def _write_summary_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _runtime_meta_path(bundle_out: Path, algo_name: str) -> Path:
    return bundle_out / "repro" / "timing" / f"{algo_name}_imputation_runtime.json"


def _load_saved_imputation_runtime(bundle_out: Path, algo_name: str) -> float | None:
    runtime_path = _runtime_meta_path(bundle_out, algo_name)
    if not runtime_path.exists():
        return None
    try:
        payload = json.loads(runtime_path.read_text())
        val = payload.get("imputation_runtime_seconds")
        return float(val) if val is not None else None
    except Exception:
        return None


def _save_imputation_runtime(bundle_out: Path, algo_name: str, runtime_seconds: float) -> None:
    runtime_path = _runtime_meta_path(bundle_out, algo_name)
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(
        runtime_path,
        {
            "algorithm": algo_name,
            "imputation_runtime_seconds": float(runtime_seconds),
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CKD EHR imputation and/or downstream classification evaluation across missingness bundles."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--algorithms", default="global_mean,row_mean,col_mean,col_mode,knn,soft_impute,tab_impute")
    parser.add_argument("--algorithm-params-json", default="{}")
    parser.add_argument(
        "--imputation-split-mode",
        choices=["row_disjoint", "transductive"],
        default="row_disjoint",
        help="row_disjoint prevents train/test leakage during imputation; transductive reproduces previous behavior.",
    )
    parser.add_argument("--stage", choices=["impute", "evaluate", "all"], default="all")
    parser.add_argument(
        "--skip-existing-imputations",
        action="store_true",
        help="During --stage impute/all, do not rerun an algorithm if <algo>_prediction.npy already exists.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun and overwrite even when output files for an algorithm already exist.",
    )
    parser.add_argument("--num-imputations", type=int, default=5)
    parser.add_argument("--mi-seed", type=int, default=7)
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument(
        "--include-mi-gaussian",
        action="store_true",
        help="Include the mi_gaussian multiple-imputation baseline in outputs.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    args = parser.parse_args()

    algorithm_names = _parse_csv_list(args.algorithms, str)
    algorithm_params = json.loads(args.algorithm_params_json)
    if not isinstance(algorithm_params, dict):
        raise ValueError("algorithm-params-json must be a JSON object")

    bundles = _discover_bundles(args.dataset_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    save_json(
        args.output_root / "run_config.json",
        {
            "dataset_root": str(args.dataset_root),
            "output_root": str(args.output_root),
            "algorithms": algorithm_names,
            "algorithm_params": algorithm_params,
            "stage": args.stage,
            "skip_existing_imputations": bool(args.skip_existing_imputations),
            "force": bool(args.force),
            "imputation_split_mode": args.imputation_split_mode,
            "num_imputations": args.num_imputations,
            "mi_seed": args.mi_seed,
            "task": args.task,
            "include_mi_gaussian": bool(args.include_mi_gaussian),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    imputation_rows: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []
    n_bundles = len(bundles)
    _progress(
        f"Processing {n_bundles} bundles with algorithms: {algorithm_names}",
        quiet=args.quiet,
    )

    for bundle_idx, bundle_dir in enumerate(bundles, start=1):
        meta = json.loads((bundle_dir / "dataset_meta.json").read_text())
        meta["bundle_id"] = bundle_dir.name
        _progress(
            f"  [{bundle_idx}/{n_bundles}] {bundle_dir.name} "
            f"(pattern={meta.get('pattern', '')}, missing={meta.get('missing_fraction', '')})",
            quiet=args.quiet,
        )

        y_true = load_matrix(bundle_dir / "ground_truth.npy")
        observed = load_matrix(bundle_dir / "observed.npy")
        eval_mask = np.load(bundle_dir / "eval_mask.npy").astype(bool)
        train_mask = np.load(bundle_dir / "downstream_train_mask.npy").astype(bool)
        test_mask = np.load(bundle_dir / "downstream_test_mask.npy").astype(bool)
        target_col = int(meta["target_col"])

        bundle_out = args.output_root / bundle_dir.name
        bundle_out.mkdir(parents=True, exist_ok=True)
        _save_bundle_snapshot(
            bundle_out=bundle_out,
            bundle_dir=bundle_dir,
            meta=meta,
            y_true=y_true,
            observed=observed,
            eval_mask=eval_mask,
            train_mask=train_mask,
            test_mask=test_mask,
        )

        if args.stage in {"impute", "all"}:
            for algo_name in algorithm_names:
                algo_cls = ALGORITHM_REGISTRY.get(algo_name)
                if algo_cls is None:
                    imputation_rows.append(
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

                pred_path = bundle_out / f"{algo_name}_prediction.npy"
                imputation_eval_path = bundle_out / f"{algo_name}_imputation_eval.json"
                already_run = pred_path.exists() and imputation_eval_path.exists()
                if already_run and not args.force:
                    _progress(f"    impute {algo_name}: skipped (already run)", quiet=args.quiet)
                    try:
                        eval_data = json.loads(imputation_eval_path.read_text())
                        imputation_rows.append(
                            _metric_summary_row(
                                bundle_meta=meta,
                                method=algo_name,
                                metrics={
                                    **{f"imputation_{k}": v for k, v in eval_data.get("imputation_metrics", {}).items()},
                                    "imputation_runtime_seconds": eval_data.get("imputation_runtime_seconds"),
                                    "status": "ok",
                                },
                            )
                        )
                    except Exception:
                        pass
                    continue
                try:
                    imputation_runtime_seconds: float | None = None
                    if args.skip_existing_imputations and pred_path.exists():
                        _progress(f"    impute {algo_name}: loading cached prediction", quiet=args.quiet)
                        pred = load_matrix(pred_path)
                        imputation_runtime_seconds = _load_saved_imputation_runtime(bundle_out, algo_name)
                    else:
                        _progress(f"    impute {algo_name}: running...", quiet=args.quiet)
                        start = time.perf_counter()
                        pred = _complete_with_split_guard(
                            algo_cls=algo_cls,
                            observed=observed,
                            y_true=y_true,
                            train_mask=train_mask,
                            test_mask=test_mask,
                            target_col=target_col,
                            split_mode=args.imputation_split_mode,
                            params=params,
                        )
                        imputation_runtime_seconds = time.perf_counter() - start
                        save_matrix(pred_path, pred)
                        _save_imputation_runtime(bundle_out, algo_name, imputation_runtime_seconds)
                    impute_metrics = evaluate_single_imputation_metrics(y_true=y_true, y_pred=pred, eval_mask=eval_mask)
                    save_json(
                        bundle_out / f"{algo_name}_imputation_eval.json",
                        {
                            "dataset_dir": str(bundle_dir),
                            "method": algo_name,
                            "target_col": target_col,
                            "imputation_metrics": impute_metrics,
                            "imputation_runtime_seconds": imputation_runtime_seconds,
                        },
                    )
                    imputation_rows.append(
                        _metric_summary_row(
                            bundle_meta=meta,
                            method=algo_name,
                            metrics={
                                **{f"imputation_{k}": v for k, v in impute_metrics.items()},
                                "imputation_runtime_seconds": imputation_runtime_seconds,
                                "status": "ok",
                            },
                        )
                    )
                    rt = imputation_runtime_seconds
                    _progress(
                        f"    impute {algo_name}: ok ({rt:.1f}s)" if rt is not None else f"    impute {algo_name}: ok",
                        quiet=args.quiet,
                    )
                except Exception as exc:
                    _progress(f"    impute {algo_name}: failed ({exc})", quiet=args.quiet)
                    imputation_rows.append(
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
                    _save_algorithm_error_artifact(
                        bundle_out=bundle_out,
                        algo_name=algo_name,
                        error_message=f"{type(exc).__name__}: {exc}",
                    )

        if args.stage in {"evaluate", "all"}:
            for algo_name in algorithm_names:
                pred_path = bundle_out / f"{algo_name}_prediction.npy"
                classification_eval_path = bundle_out / f"{algo_name}_classification_eval.json"
                if not pred_path.exists():
                    _progress(f"    evaluate {algo_name}: skipped (missing prediction)", quiet=args.quiet)
                    classification_rows.append(
                        {
                            "bundle": bundle_dir.name,
                            "pattern": meta.get("pattern", ""),
                            "missing_fraction": meta.get("missing_fraction", ""),
                            "seed": meta.get("seed", ""),
                            "method": algo_name,
                            "status": "failed",
                            "error": f"missing prediction file: {pred_path}",
                        }
                    )
                    continue
                already_run = classification_eval_path.exists()
                if already_run and not args.force:
                    _progress(f"    evaluate {algo_name}: skipped (already run)", quiet=args.quiet)
                    try:
                        eval_data = json.loads(classification_eval_path.read_text())
                        impute_metrics = eval_data.get("imputation_metrics", {})
                        down_metrics = eval_data.get("downstream_metrics", {})
                        classification_rows.append(
                            _metric_summary_row(
                                bundle_meta=meta,
                                method=algo_name,
                                metrics={
                                    **{f"imputation_{k}": v for k, v in impute_metrics.items()},
                                    **down_metrics,
                                    "imputation_runtime_seconds": eval_data.get("imputation_runtime_seconds"),
                                    "downstream_runtime_seconds": eval_data.get("downstream_runtime_seconds"),
                                    "status": "ok",
                                },
                            )
                        )
                    except Exception:
                        pass
                    continue
                try:
                    _progress(f"    evaluate {algo_name}: running...", quiet=args.quiet)
                    pred = load_matrix(pred_path)
                    impute_metrics = evaluate_single_imputation_metrics(y_true=y_true, y_pred=pred, eval_mask=eval_mask)
                    imputation_runtime_seconds = _load_saved_imputation_runtime(bundle_out, algo_name)
                    start = time.perf_counter()
                    downstream_payload = evaluate_downstream_models_with_predictions(
                        y_true=y_true,
                        y_pred=pred,
                        target_col=target_col,
                        train_row_mask=train_mask,
                        test_row_mask=test_mask,
                        task=args.task,
                    )
                    downstream_runtime_seconds = time.perf_counter() - start
                    down_metrics = dict(downstream_payload["scores"])
                    payload = {
                        "dataset_dir": str(bundle_dir),
                        "method": algo_name,
                        "task": args.task,
                        "target_col": target_col,
                        "prediction_path": str(pred_path),
                        "imputation_runtime_seconds": imputation_runtime_seconds,
                        "downstream_runtime_seconds": downstream_runtime_seconds,
                        "imputation_metrics": impute_metrics,
                        "downstream_metrics": down_metrics,
                    }
                    save_json(bundle_out / f"{algo_name}_classification_eval.json", payload)
                    _save_algorithm_task_artifacts(
                        bundle_out=bundle_out,
                        algo_name=algo_name,
                        y_pred=pred,
                        downstream_payload=downstream_payload,
                    )
                    classification_rows.append(
                        _metric_summary_row(
                            bundle_meta=meta,
                            method=algo_name,
                            metrics={
                                **{f"imputation_{k}": v for k, v in impute_metrics.items()},
                                **down_metrics,
                                "imputation_runtime_seconds": imputation_runtime_seconds,
                                "downstream_runtime_seconds": downstream_runtime_seconds,
                                "status": "ok",
                            },
                        )
                    )
                    _progress(
                        f"    evaluate {algo_name}: ok ({downstream_runtime_seconds:.1f}s)",
                        quiet=args.quiet,
                    )
                except Exception as exc:
                    _progress(f"    evaluate {algo_name}: failed ({exc})", quiet=args.quiet)
                    classification_rows.append(
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
                    _save_algorithm_error_artifact(
                        bundle_out=bundle_out,
                        algo_name=algo_name,
                        error_message=f"{type(exc).__name__}: {exc}",
                    )

        if args.include_mi_gaussian:
            if args.stage in {"impute", "all"}:
                mi_pred_paths = [bundle_out / f"mi_gaussian_prediction_{idx:03d}.npy" for idx in range(args.num_imputations)]
                mi_impute_already_run = all(p.exists() for p in mi_pred_paths)
                if mi_impute_already_run and not args.force:
                    _progress("    impute mi_gaussian: skipped (already run)", quiet=args.quiet)
                    imputation_rows.append(
                        _metric_summary_row(
                            bundle_meta=meta,
                            method="mi_gaussian",
                            metrics={"num_imputations": float(args.num_imputations), "status": "ok"},
                        )
                    )
                else:
                    try:
                        _progress("    impute mi_gaussian: running...", quiet=args.quiet)
                        mi_preds = generate_multiple_imputations_gaussian(
                            observed=observed,
                            num_imputations=args.num_imputations,
                            seed=args.mi_seed,
                        )
                        for idx, pred in enumerate(mi_preds):
                            save_matrix(bundle_out / f"mi_gaussian_prediction_{idx:03d}.npy", pred)
                        imputation_rows.append(
                            _metric_summary_row(
                                bundle_meta=meta,
                                method="mi_gaussian",
                                metrics={"num_imputations": float(len(mi_preds)), "status": "ok"},
                            )
                        )
                        _progress("    impute mi_gaussian: ok", quiet=args.quiet)
                    except Exception as exc:
                        _progress(f"    impute mi_gaussian: failed ({exc})", quiet=args.quiet)
                        imputation_rows.append(
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
            if args.stage in {"evaluate", "all"}:
                mi_classification_eval_path = bundle_out / "mi_gaussian_classification_eval.json"
                if mi_classification_eval_path.exists() and not args.force:
                    _progress("    evaluate mi_gaussian: skipped (already run)", quiet=args.quiet)
                    try:
                        eval_data = json.loads(mi_classification_eval_path.read_text())
                        mi_metrics = eval_data.get("multiple_imputation_metrics", {})
                        classification_rows.append(
                            _metric_summary_row(
                                bundle_meta=meta,
                                method="mi_gaussian",
                                metrics={**mi_metrics, "status": "ok"},
                            )
                        )
                    except Exception:
                        pass
                else:
                    try:
                        _progress("    evaluate mi_gaussian: running...", quiet=args.quiet)
                        mi_preds = []
                        for idx in range(args.num_imputations):
                            pred_path = bundle_out / f"mi_gaussian_prediction_{idx:03d}.npy"
                            if not pred_path.exists():
                                raise FileNotFoundError(f"missing prediction file: {pred_path}")
                            mi_preds.append(load_matrix(pred_path))
                        mi_metrics = evaluate_multiple_imputation_metrics(
                            y_true=y_true,
                            y_preds=mi_preds,
                            eval_mask=eval_mask,
                            target_col=target_col,
                            train_row_mask=train_mask,
                            test_row_mask=test_mask,
                            task=args.task,
                        )
                        save_json(
                            bundle_out / "mi_gaussian_classification_eval.json",
                            {
                                "dataset_dir": str(bundle_dir),
                                "method": "mi_gaussian",
                                "task": args.task,
                                "target_col": target_col,
                                "num_imputations": args.num_imputations,
                                "multiple_imputation_metrics": mi_metrics,
                            },
                        )
                        classification_rows.append(
                            _metric_summary_row(
                                bundle_meta=meta,
                                method="mi_gaussian",
                                metrics={**mi_metrics, "status": "ok"},
                            )
                        )
                        _progress("    evaluate mi_gaussian: ok", quiet=args.quiet)
                    except Exception as exc:
                        _progress(f"    evaluate mi_gaussian: failed ({exc})", quiet=args.quiet)
                        classification_rows.append(
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

    if args.stage in {"impute", "all"}:
        impute_summary_csv = args.output_root / "ckd_ehr_imputation_summary.csv"
        _write_summary_csv(imputation_rows, impute_summary_csv)
        print(f"wrote CKD EHR imputation summary: {impute_summary_csv}")
    if args.stage in {"evaluate", "all"}:
        classification_summary_csv = args.output_root / "ckd_ehr_classification_summary.csv"
        _write_summary_csv(classification_rows, classification_summary_csv)
        print(f"wrote CKD EHR classification summary: {classification_summary_csv}")


if __name__ == "__main__":
    main()
