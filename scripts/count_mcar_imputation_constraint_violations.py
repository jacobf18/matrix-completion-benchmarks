#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Count out-of-range imputed values for each method across MCAR bundles, "
            "using column constraints and bundle eval masks."
        )
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("benchmarks/reports/ckd_ehr_all_algorithms"),
        help="Report root containing pattern_mcar* bundle directories and *_prediction.npy files.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("benchmarks/datasets/ckd_ehr_all_algorithms"),
        help="Dataset root containing source_meta.json for feature column names.",
    )
    parser.add_argument(
        "--constraints-json",
        type=Path,
        default=Path("benchmarks/sources/ckd_ehr_abu_dhabi/numeric_column_constraints.json"),
        help="JSON file with per-column lower/upper constraints.",
    )
    parser.add_argument(
        "--pattern-prefix",
        type=str,
        default="pattern_mcar",
        help="Bundle directory prefix to scan (default: pattern_mcar).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("benchmarks/reports/ckd_ehr_all_algorithms/mcar_constraint_violations_by_method.csv"),
        help="Output CSV path for method-level counts.",
    )
    parser.add_argument(
        "--output-column-csv",
        type=Path,
        default=Path("benchmarks/reports/ckd_ehr_all_algorithms/mcar_constraint_violations_by_method_and_column.csv"),
        help="Output CSV path for method+column counts.",
    )
    return parser


def _load_constraints(path: Path) -> dict[str, tuple[float | None, float | None]]:
    payload = json.loads(path.read_text())
    raw = payload.get("constraints", {})
    out: dict[str, tuple[float | None, float | None]] = {}
    for col, spec in raw.items():
        lower = spec.get("lower")
        upper = spec.get("upper")
        out[str(col)] = (None if lower is None else float(lower), None if upper is None else float(upper))
    return out


def _load_column_names(dataset_root: Path, target_fallback: str = "EventCKD35") -> list[str]:
    source_meta_path = dataset_root / "source_meta.json"
    if not source_meta_path.exists():
        raise FileNotFoundError(f"Missing source_meta.json under dataset root: {source_meta_path}")
    meta = json.loads(source_meta_path.read_text())
    feature_cols = [str(c) for c in meta.get("feature_columns", [])]
    target = str(meta.get("target_column", target_fallback))
    return [*feature_cols, target]


def _iter_mcar_bundles(report_root: Path, pattern_prefix: str) -> list[Path]:
    bundles = sorted(
        p for p in report_root.iterdir() if p.is_dir() and p.name.startswith(pattern_prefix) and (p / "dataset_meta.json").exists()
    )
    if not bundles:
        # Fallback: some report bundles may not contain dataset_meta.json
        bundles = sorted(
            p for p in report_root.iterdir() if p.is_dir() and p.name.startswith(pattern_prefix) and (p / "repro" / "bundle_snapshot").exists()
        )
    return bundles


def main() -> None:
    args = build_parser().parse_args()

    if not args.report_root.exists():
        raise FileNotFoundError(f"report root not found: {args.report_root}")
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {args.dataset_root}")
    if not args.constraints_json.exists():
        raise FileNotFoundError(f"constraints file not found: {args.constraints_json}")

    constraints = _load_constraints(args.constraints_json)
    col_names = _load_column_names(args.dataset_root)
    bundles = _iter_mcar_bundles(args.report_root, args.pattern_prefix)
    if not bundles:
        raise ValueError(f"No bundles found with prefix '{args.pattern_prefix}' under {args.report_root}")

    constrained_col_idx: dict[int, tuple[str, float | None, float | None]] = {}
    for idx, col in enumerate(col_names):
        if col in constraints:
            lo, hi = constraints[col]
            constrained_col_idx[idx] = (col, lo, hi)

    if not constrained_col_idx:
        raise ValueError("No constrained columns matched matrix column names.")

    by_method: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_method_col: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))

    bundle_count = 0
    for bundle in bundles:
        snap_dir = bundle / "repro" / "bundle_snapshot"
        eval_mask_path = snap_dir / "eval_mask.npy"
        if not eval_mask_path.exists():
            continue
        eval_mask = np.load(eval_mask_path).astype(bool)

        pred_files = sorted(bundle.glob("*_prediction.npy"))
        if not pred_files:
            continue
        bundle_count += 1

        for pred_path in pred_files:
            method = pred_path.stem[: -len("_prediction")]
            pred = np.load(pred_path)
            if pred.shape != eval_mask.shape:
                continue

            total_checked = 0
            total_nan = 0
            total_oob = 0

            for c_idx, (c_name, lo, hi) in constrained_col_idx.items():
                imputed_col_mask = eval_mask[:, c_idx]
                if not np.any(imputed_col_mask):
                    continue
                vals = pred[:, c_idx][imputed_col_mask]
                n = int(vals.size)
                if n == 0:
                    continue
                nan_mask = ~np.isfinite(vals)
                n_nan = int(np.sum(nan_mask))
                finite_vals = vals[~nan_mask]
                n_oob = 0
                if finite_vals.size > 0:
                    if lo is not None:
                        n_oob += int(np.sum(finite_vals < lo))
                    if hi is not None:
                        n_oob += int(np.sum(finite_vals > hi))

                total_checked += n
                total_nan += n_nan
                total_oob += n_oob

                key = (method, c_name)
                by_method_col[key]["checked_count"] += n
                by_method_col[key]["nan_count"] += n_nan
                by_method_col[key]["out_of_bounds_count"] += n_oob

            by_method[method]["checked_count"] += total_checked
            by_method[method]["nan_count"] += total_nan
            by_method[method]["out_of_bounds_count"] += total_oob

    if bundle_count == 0:
        raise ValueError("No valid bundles found with eval masks and prediction files.")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "method",
            "checked_count",
            "nan_count",
            "out_of_bounds_count",
            "nan_rate",
            "out_of_bounds_rate",
            "valid_out_of_bounds_rate",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for method in sorted(by_method.keys()):
            checked = int(by_method[method]["checked_count"])
            nan_count = int(by_method[method]["nan_count"])
            oob_count = int(by_method[method]["out_of_bounds_count"])
            valid = max(checked - nan_count, 0)
            writer.writerow(
                {
                    "method": method,
                    "checked_count": checked,
                    "nan_count": nan_count,
                    "out_of_bounds_count": oob_count,
                    "nan_rate": (nan_count / checked) if checked else "",
                    "out_of_bounds_rate": (oob_count / checked) if checked else "",
                    "valid_out_of_bounds_rate": (oob_count / valid) if valid else "",
                }
            )

    args.output_column_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_column_csv.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "method",
            "column",
            "checked_count",
            "nan_count",
            "out_of_bounds_count",
            "nan_rate",
            "out_of_bounds_rate",
            "valid_out_of_bounds_rate",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for method, column in sorted(by_method_col.keys()):
            checked = int(by_method_col[(method, column)]["checked_count"])
            nan_count = int(by_method_col[(method, column)]["nan_count"])
            oob_count = int(by_method_col[(method, column)]["out_of_bounds_count"])
            valid = max(checked - nan_count, 0)
            writer.writerow(
                {
                    "method": method,
                    "column": column,
                    "checked_count": checked,
                    "nan_count": nan_count,
                    "out_of_bounds_count": oob_count,
                    "nan_rate": (nan_count / checked) if checked else "",
                    "out_of_bounds_rate": (oob_count / checked) if checked else "",
                    "valid_out_of_bounds_rate": (oob_count / valid) if valid else "",
                }
            )

    print(f"bundles_scanned: {bundle_count}")
    print(f"wrote method summary: {args.output_csv}")
    print(f"wrote method+column summary: {args.output_column_csv}")


if __name__ == "__main__":
    main()
