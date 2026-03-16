#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def _parse_csv_list(raw: str) -> list[str]:
    vals = [token.strip() for token in raw.split(",") if token.strip()]
    if not vals:
        raise ValueError("Expected at least one comma-separated value.")
    return vals


def _pretty_text(raw: str) -> str:
    text = str(raw).replace("_", " ").strip()
    words = [w.capitalize() for w in text.split()]
    return " ".join(words)


def _pretty_method(method: str) -> str:
    if method == "tab_impute":
        return "TabImpute"
    return _pretty_text(method)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot CKD benchmark metrics for a given missingness pattern as barplot subplots "
            "(one subplot per metric, bars = algorithms)."
        )
    )
    parser.add_argument(
        "--classification-csv",
        type=Path,
        default=Path("/tmp/ckd_ehr_censor_reports_det/ckd_ehr_classification_summary.csv"),
        help="Classification summary CSV produced by run_ckd_ehr_classification_benchmark.py.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="ckd_egfr_censor_gt100",
        help="Pattern name to filter on (e.g., ckd_egfr_censor_gt100).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to the same directory as --classification-csv.",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=4,
        help="Number of subplot columns.",
    )
    parser.add_argument(
        "--figscale",
        type=float,
        default=3.6,
        help="Per-subplot size multiplier in inches.",
    )
    parser.add_argument(
        "--include-oracle",
        action="store_true",
        help="Add an oracle bar computed from fully observed data in report snapshots.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="",
        help=(
            "Optional comma-separated metric columns (e.g. "
            "downstream_f1_xgboost,imputation_nrmse,imputation_runtime_seconds). "
            "Default: all numeric metrics."
        ),
    )
    return parser


def _is_metric_column(name: str) -> bool:
    non_metrics = {"bundle", "method", "missing_fraction", "pattern", "seed", "status"}
    return name not in non_metrics


def _compute_oracle_rows(
    report_root: Path,
    rows_for_pattern: list[dict[str, str]],
    task: str,
) -> list[dict[str, str]]:
    import json
    import sys

    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from mcbench.workflows.tabular import evaluate_downstream_models_with_predictions, evaluate_single_imputation_metrics

    oracle_rows: list[dict[str, str]] = []
    bundle_names = sorted({str(r.get("bundle", "")) for r in rows_for_pattern if str(r.get("bundle", ""))})
    if not bundle_names:
        return oracle_rows

    for bundle_name in bundle_names:
        bundle_dir = report_root / bundle_name
        snap_dir = bundle_dir / "repro" / "bundle_snapshot"
        if not snap_dir.exists():
            continue
        meta = json.loads((snap_dir / "snapshot_meta.json").read_text())
        bundle_meta = dict(meta.get("bundle_meta", {}))
        target_col = int(bundle_meta["target_col"])
        y_true = np.load(snap_dir / "ground_truth.npy")
        eval_mask = np.load(snap_dir / "eval_mask.npy").astype(bool)
        train_mask = np.load(snap_dir / "downstream_train_mask.npy").astype(bool)
        test_mask = np.load(snap_dir / "downstream_test_mask.npy").astype(bool)

        impute_scores = evaluate_single_imputation_metrics(y_true=y_true, y_pred=y_true, eval_mask=eval_mask)
        t0 = time.perf_counter()
        downstream_payload = evaluate_downstream_models_with_predictions(
            y_true=y_true,
            y_pred=y_true,
            target_col=target_col,
            train_row_mask=train_mask,
            test_row_mask=test_mask,
            task=task,
        )
        downstream_runtime = time.perf_counter() - t0
        downstream_scores = dict(downstream_payload["scores"])
        for key, val in list(downstream_scores.items()):
            if key.startswith("downstream_average_precision_"):
                pr_auc_key = key.replace("downstream_average_precision_", "downstream_pr_auc_", 1)
                downstream_scores[pr_auc_key] = val

        row: dict[str, str] = {
            "bundle": bundle_name,
            "method": "oracle",
            "missing_fraction": str(bundle_meta.get("missing_fraction", "")),
            "pattern": str(bundle_meta.get("pattern", "")),
            "seed": str(bundle_meta.get("seed", "")),
            "status": "ok",
            "imputation_rmse": str(float(impute_scores["rmse"])),
            "imputation_mae": str(float(impute_scores["mae"])),
            "imputation_nrmse": str(float(impute_scores["nrmse"])),
            "imputation_nmae": str(float(impute_scores["nmae"])),
            "imputation_runtime_seconds": "0.0",
            "downstream_runtime_seconds": str(float(downstream_runtime)),
        }
        for k, v in downstream_scores.items():
            row[str(k)] = str(float(v))
        oracle_rows.append(row)
    return oracle_rows


def main() -> None:
    args = build_parser().parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_ckd_pattern_metrics_bars.py requires matplotlib. Install with: pip install matplotlib"
        ) from exc
    plt.rcParams["font.family"] = "serif"

    if not args.classification_csv.exists():
        raise FileNotFoundError(f"Classification CSV not found: {args.classification_csv}")
    if args.output_path is None:
        args.output_path = args.classification_csv.parent / f"{args.pattern}_metrics_barplots.png"

    with args.classification_csv.open("r", newline="", encoding="utf-8") as fh:
        rows = [row for row in csv.DictReader(fh)]
    if not rows:
        raise ValueError(f"No rows found in: {args.classification_csv}")

    pattern_rows = [row for row in rows if str(row.get("pattern", "")) == args.pattern]
    if not pattern_rows:
        raise ValueError(f"No rows found for pattern='{args.pattern}' in: {args.classification_csv}")
    inferred_task = (
        "classification"
        if any("downstream_accuracy_" in key for key in pattern_rows[0].keys())
        else "regression"
    )
    if args.include_oracle:
        pattern_rows = [*pattern_rows, *_compute_oracle_rows(args.classification_csv.parent, pattern_rows, inferred_task)]

    if args.metrics.strip():
        metric_names = _parse_csv_list(args.metrics)
    else:
        metric_names = [c for c in pattern_rows[0].keys() if _is_metric_column(c)]
    if not metric_names:
        raise ValueError("No metric columns found to plot.")

    by_method: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in pattern_rows:
        method = str(row.get("method", ""))
        if not method:
            continue
        for metric in metric_names:
            raw = str(row.get(metric, "")).strip()
            if raw == "":
                continue
            try:
                by_method[method][metric].append(float(raw))
            except ValueError:
                continue

    methods = sorted(by_method.keys())
    if not methods:
        raise ValueError("No methods with numeric metrics found for requested pattern.")
    display_methods = [_pretty_method(m) for m in methods]

    metric_to_values: dict[str, list[float]] = {}
    for metric in metric_names:
        vals = []
        for method in methods:
            samples = by_method[method].get(metric, [])
            vals.append(sum(samples) / len(samples) if samples else float("nan"))
        if any(not math.isnan(v) for v in vals):
            metric_to_values[metric] = vals

    metrics = sorted(metric_to_values.keys())
    n_metrics = len(metrics)
    ncols = max(1, args.ncols)
    nrows = math.ceil(n_metrics / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(args.figscale * ncols, args.figscale * nrows),
        squeeze=False,
    )
    x = list(range(len(methods)))
    for idx, metric in enumerate(metrics):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        y = metric_to_values[metric]
        ax.bar(x, y)
        ax.set_title(_pretty_text(metric), fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(display_methods, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)

    for idx in range(n_metrics, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output_path, dpi=180)
    print(f"wrote plot: {args.output_path}")


if __name__ == "__main__":
    main()
