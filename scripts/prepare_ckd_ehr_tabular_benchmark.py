#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mcbench.datasets.missingness import apply_missingness, generate_missingness_mask
from mcbench.io import save_json, save_mask, save_matrix


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


def _load_and_encode_csv(input_csv: Path, target_column: str) -> tuple[np.ndarray, list[str], int]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "prepare_ckd_ehr_tabular_benchmark.py requires pandas. Install with: pip install pandas"
        ) from exc

    df = pd.read_csv(input_csv)
    if target_column not in df.columns:
        known = ", ".join(str(c) for c in df.columns)
        raise ValueError(f"target-column '{target_column}' not found. Known: {known}")

    missing_tokens = {"", "na", "n/a", "null", "none", "?", "nan"}
    cleaned = df.copy()
    for col in cleaned.columns:
        series = cleaned[col]
        if series.dtype == object:
            normalized = series.astype(str).str.strip()
            normalized = normalized.mask(normalized.str.lower().isin(missing_tokens))
            cleaned[col] = normalized.replace("nan", np.nan)

    # Target is regression-focused; coerce to numeric and drop rows with missing target.
    target = np.asarray(pd.to_numeric(cleaned[target_column], errors="coerce"), dtype=np.float64)
    valid_target = np.isfinite(target)
    cleaned = cleaned.loc[valid_target].reset_index(drop=True)
    target = target[valid_target]

    feature_cols = [c for c in cleaned.columns if c != target_column]
    encoded_features = []
    encoded_feature_names = []

    for col in feature_cols:
        series = cleaned[col]
        numeric = pd.to_numeric(series, errors="coerce")
        finite_numeric = int(np.isfinite(np.asarray(numeric)).sum())
        finite_total = int(series.notna().sum())

        if finite_total > 0 and finite_numeric / finite_total >= 0.9:
            encoded_features.append(np.asarray(numeric, dtype=np.float64).reshape(-1, 1))
            encoded_feature_names.append(str(col))
            continue

        dummies = pd.get_dummies(series, prefix=str(col), dummy_na=False)
        if dummies.shape[1] == 0:
            # All missing categorical column: keep one NaN column to preserve schema.
            encoded_features.append(np.full((series.shape[0], 1), np.nan, dtype=np.float64))
            encoded_feature_names.append(f"{col}__missing")
        else:
            encoded_features.append(np.asarray(dummies, dtype=np.float64))
            encoded_feature_names.extend([str(c) for c in dummies.columns])

    if not encoded_features:
        raise ValueError("No feature columns after encoding.")

    x = np.concatenate(encoded_features, axis=1)
    y = target.reshape(-1, 1)
    full = np.concatenate([x, y], axis=1)

    # Ground truth for imputation metrics should be fully observed.
    keep = np.all(np.isfinite(full), axis=1)
    if int(np.sum(keep)) < 10:
        raise ValueError(
            "Too few fully observed rows after preprocessing (<10). "
            "Try a different target or preprocessing strategy."
        )

    full = full[keep]
    col_names = [*encoded_feature_names, target_column]
    target_idx = full.shape[1] - 1
    return full.astype(np.float64), col_names, target_idx


def _default_excluded_columns(target_column: str) -> set[str]:
    if target_column == "TimeToEventMonths":
        # Prevent direct horizon leakage when predicting time-to-event.
        return {"TIME_YEAR", "EventCKD35"}
    if target_column == "EventCKD35":
        # Prevent post-outcome horizon leakage for event classification.
        return {"TimeToEventMonths", "TIME_YEAR"}
    return set()


def _make_eval_mask(
    matrix: np.ndarray,
    target_col: int,
    pattern: str,
    missing_fraction: float,
    seed: int,
) -> np.ndarray:
    feature_matrix = matrix[:, :target_col]
    feature_mask = generate_missingness_mask(
        matrix=feature_matrix,
        kind=pattern,
        missing_fraction=missing_fraction,
        seed=seed,
    )
    eval_mask = np.zeros_like(matrix, dtype=bool)
    eval_mask[:, :target_col] = feature_mask
    return eval_mask


def _train_test_split(n_rows: int, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_rows)
    n_test = max(1, int(round(n_rows * test_fraction)))
    test_idx = order[:n_test]
    train_idx = order[n_test:]

    train_mask = np.zeros(n_rows, dtype=bool)
    test_mask = np.zeros(n_rows, dtype=bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    return train_mask, test_mask


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare CKD EHR tabular benchmark bundles with configurable missingness patterns "
            "for downstream regression and imputation metrics."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-csv", type=Path)
    input_group.add_argument(
        "--source-dir",
        type=Path,
        help="Directory containing one or more CSV files (largest CSV will be used).",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--target-column", required=True, help="Regression target column name in CSV.")
    parser.add_argument(
        "--exclude-columns",
        default="",
        help="Comma-separated columns to drop from features before encoding.",
    )
    parser.add_argument("--patterns", default="mcar,mar_logistic,mnar_self_logistic,block,bursty")
    parser.add_argument("--missing-fractions", default="0.1,0.2,0.4")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args()

    patterns = _parse_csv_list(args.patterns, str)
    fractions = _parse_csv_list(args.missing_fractions, float)
    seeds = _parse_csv_list(args.seeds, int)

    for f in fractions:
        if not (0 < f < 1):
            raise ValueError("All missing-fractions must be in (0, 1).")
    if not (0 < args.test_fraction < 1):
        raise ValueError("test-fraction must be in (0,1).")

    input_csv: Path
    if args.input_csv is not None:
        input_csv = args.input_csv
    else:
        csv_candidates = sorted(args.source_dir.rglob("*.csv")) if args.source_dir is not None else []
        if not csv_candidates:
            raise ValueError(f"No CSV files found under source-dir: {args.source_dir}")
        input_csv = max(csv_candidates, key=lambda p: p.stat().st_size)

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "prepare_ckd_ehr_tabular_benchmark.py requires pandas. Install with: pip install pandas"
        ) from exc

    user_excluded = {c.strip() for c in args.exclude_columns.split(",") if c.strip()}
    auto_excluded = _default_excluded_columns(args.target_column)
    excluded = user_excluded | auto_excluded

    df = pd.read_csv(input_csv)
    missing_excluded = sorted(col for col in excluded if col not in df.columns)
    if missing_excluded:
        # Ignore absent excludes; keep behavior robust across dataset revisions.
        excluded = {c for c in excluded if c in df.columns}
    drop_cols = [c for c in sorted(excluded) if c != args.target_column]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    tmp_encoded_csv = args.output_root / "_tmp_ckd_encoded_source.csv"
    args.output_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp_encoded_csv, index=False)

    matrix, col_names, target_col = _load_and_encode_csv(tmp_encoded_csv, args.target_column)
    try:
        tmp_encoded_csv.unlink(missing_ok=True)
    except Exception:
        pass

    args.output_root.mkdir(parents=True, exist_ok=True)
    save_matrix(args.output_root / "full_matrix.npy", matrix)
    save_json(
        args.output_root / "source_meta.json",
        {
            "input_csv": str(input_csv),
            "target_column": args.target_column,
            "target_col_index": target_col,
            "excluded_columns": sorted(drop_cols),
            "n_rows": int(matrix.shape[0]),
            "n_cols": int(matrix.shape[1]),
            "patterns": patterns,
            "missing_fractions": fractions,
            "seeds": seeds,
            "feature_columns": col_names[:target_col],
        },
    )

    for pattern in patterns:
        for fraction in fractions:
            for seed in seeds:
                bundle_id = f"pattern_{pattern}__frac_{fraction:.3f}__seed_{seed}".replace(".", "p")
                out_dir = args.output_root / bundle_id
                eval_mask = _make_eval_mask(
                    matrix=matrix,
                    target_col=target_col,
                    pattern=pattern,
                    missing_fraction=fraction,
                    seed=seed,
                )
                observed = apply_missingness(matrix=matrix, missing_mask=eval_mask)
                train_mask, test_mask = _train_test_split(
                    n_rows=matrix.shape[0], test_fraction=args.test_fraction, seed=seed
                )

                out_dir.mkdir(parents=True, exist_ok=True)
                save_matrix(out_dir / "ground_truth.npy", matrix)
                save_matrix(out_dir / "observed.npy", observed)
                save_mask(out_dir / "eval_mask.npy", eval_mask)
                save_mask(out_dir / "downstream_train_mask.npy", train_mask)
                save_mask(out_dir / "downstream_test_mask.npy", test_mask)
                save_json(
                    out_dir / "dataset_meta.json",
                    {
                        "task": "regression",
                        "pattern": pattern,
                        "missing_fraction": float(fraction),
                        "seed": int(seed),
                        "target_col": int(target_col),
                        "target_column": args.target_column,
                        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
                        "eval_count": int(np.sum(eval_mask)),
                        "eval_fraction_actual": float(np.sum(eval_mask) / np.sum(np.isfinite(matrix[:, :target_col]))),
                    },
                )

    print(f"wrote CKD EHR tabular bundles: {args.output_root}")


if __name__ == "__main__":
    main()
