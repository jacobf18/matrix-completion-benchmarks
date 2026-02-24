#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mcbench.datasets.missingness import apply_missingness, generate_missingness_mask
from mcbench.io import load_matrix, save_json, save_mask, save_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reusable missingness masks and observed matrices.")
    parser.add_argument("--input-matrix", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--pattern",
        choices=["mcar", "mar_logistic", "mnar_self_logistic", "block", "bursty"],
        default="mcar",
    )
    parser.add_argument("--missing-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-col", type=int, default=None, help="Used for mar_logistic.")
    parser.add_argument("--block-axis", choices=["rows", "cols"], default="rows")
    parser.add_argument("--burst-max", type=int, default=12)
    args = parser.parse_args()

    matrix = load_matrix(args.input_matrix)
    missing_mask = generate_missingness_mask(
        matrix=matrix,
        kind=args.pattern,
        missing_fraction=args.missing_fraction,
        seed=args.seed,
        feature_col=args.feature_col,
        block_axis=args.block_axis,
        burst_max=args.burst_max,
    )
    observed = apply_missingness(matrix=matrix, missing_mask=missing_mask)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_matrix(args.output_dir / "ground_truth.npy", matrix)
    save_matrix(args.output_dir / "observed.npy", observed)
    save_mask(args.output_dir / "missing_mask.npy", missing_mask)
    save_mask(args.output_dir / "observed_mask.npy", ~missing_mask)
    save_json(
        args.output_dir / "missingness_meta.json",
        {
            "input_matrix": str(args.input_matrix),
            "pattern": args.pattern,
            "missing_fraction_requested": args.missing_fraction,
            "missing_fraction_actual": float(np.sum(missing_mask) / np.sum(np.isfinite(matrix))),
            "seed": args.seed,
            "feature_col": args.feature_col,
            "block_axis": args.block_axis,
            "burst_max": args.burst_max,
        },
    )
    print(f"wrote missingness bundle: {args.output_dir}")


if __name__ == "__main__":
    main()

