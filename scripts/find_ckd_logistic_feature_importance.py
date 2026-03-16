#!/usr/bin/env python3
"""Fit logistic regression on a CKD benchmark bundle and rank feature importance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train logistic regression on a CKD benchmark bundle and rank input "
            "columns by absolute coefficient magnitude."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("benchmarks/datasets/ckd_ehr_all_algorithms"),
        help="Root directory containing bundle folders.",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help=(
            "Specific bundle directory containing x_train.npy/y_train.npy. "
            "If omitted, the first bundle under --dataset-root is used."
        ),
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of top features to print.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save ranked importances as JSON.",
    )
    return parser


def resolve_bundle_dir(dataset_root: Path, bundle_dir: Path | None) -> Path:
    if bundle_dir is not None:
        resolved = bundle_dir.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"bundle directory does not exist: {resolved}")
        return resolved

    candidates = sorted(p for p in dataset_root.iterdir() if p.is_dir() and (p / "x_train.npy").exists())
    if not candidates:
        raise FileNotFoundError(f"no bundle directories with x_train.npy found under: {dataset_root}")
    return candidates[0]


def load_meta(meta_path: Path) -> dict[str, Any]:
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def main() -> None:
    args = build_parser().parse_args()
    bundle_dir = resolve_bundle_dir(args.dataset_root, args.bundle_dir)

    x_train = np.load(bundle_dir / "x_train.npy")
    y_train = np.load(bundle_dir / "y_train.npy")
    x_test_path = bundle_dir / "x_test.npy"
    y_test_path = bundle_dir / "y_test.npy"
    feature_cols = np.load(bundle_dir / "feature_cols.npy")
    meta = load_meta(bundle_dir / "dataset_meta.json")

    y_unique = np.unique(y_train[~np.isnan(y_train)])
    if y_unique.size != 2:
        raise ValueError(
            f"logistic regression requires a binary target; found {y_unique.size} unique values: {y_unique}"
        )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, solver="lbfgs")),
        ]
    )
    model.fit(x_train, y_train)

    coefficients = model.named_steps["clf"].coef_.ravel()
    abs_coefficients = np.abs(coefficients)
    ranked_idx = np.argsort(abs_coefficients)[::-1]
    top_k = min(args.top_k, ranked_idx.size)

    print(f"bundle_dir: {bundle_dir}")
    if meta:
        target_name = meta.get("target_column", "<unknown>")
        target_col = meta.get("target_col", "<unknown>")
        print(f"target: {target_name} (col {target_col})")

    if x_test_path.exists() and y_test_path.exists():
        x_test = np.load(x_test_path)
        y_test = np.load(y_test_path)
        y_test_unique = np.unique(y_test[~np.isnan(y_test)])
        if y_test_unique.size == 2:
            y_score = model.predict_proba(x_test)[:, 1]
            auc = roc_auc_score(y_test, y_score)
            print(f"test_auc: {auc:.6f}")

    print("rank,feature_position,original_column,coefficient,abs_coefficient")
    rankings: list[dict[str, Any]] = []
    for rank, idx in enumerate(ranked_idx[:top_k], start=1):
        original_col = int(feature_cols[idx])
        coef = float(coefficients[idx])
        abs_coef = float(abs_coefficients[idx])
        print(f"{rank},{idx},{original_col},{coef:.6f},{abs_coef:.6f}")
        rankings.append(
            {
                "rank": rank,
                "feature_position": int(idx),
                "original_column": original_col,
                "coefficient": coef,
                "abs_coefficient": abs_coef,
            }
        )

    best = rankings[0]
    print(
        "most_important_feature: "
        f"feature_position={best['feature_position']}, original_column={best['original_column']}"
    )

    if args.output_json is not None:
        payload = {
            "bundle_dir": str(bundle_dir),
            "target_column": meta.get("target_column"),
            "target_col": meta.get("target_col"),
            "rankings": rankings,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"wrote: {args.output_json}")


if __name__ == "__main__":
    main()
