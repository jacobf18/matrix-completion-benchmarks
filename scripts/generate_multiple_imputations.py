#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mcbench.io import load_matrix, save_json, save_matrix
from mcbench.workflows.tabular import generate_multiple_imputations_gaussian


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multiple imputations from observed tabular matrix.")
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--num-imputations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    observed = load_matrix(args.dataset_dir / "observed.npy")
    imputations = generate_multiple_imputations_gaussian(
        observed=observed,
        num_imputations=args.num_imputations,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, matrix in enumerate(imputations):
        path = args.output_dir / f"prediction_{idx:03d}.npy"
        save_matrix(path, matrix)
        paths.append(str(path))

    save_json(
        args.output_dir / "manifest.json",
        {
            "dataset_dir": str(args.dataset_dir),
            "num_imputations": args.num_imputations,
            "seed": args.seed,
            "prediction_paths": paths,
        },
    )
    print(f"wrote multiple imputations to: {args.output_dir}")
    print(f"count: {len(paths)}")


if __name__ == "__main__":
    main()

