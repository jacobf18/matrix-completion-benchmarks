#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from mcbench.workflows.tabular import prepare_tabular_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a tabular missing-data benchmark bundle.")
    parser.add_argument("--input-matrix", required=True, type=Path)
    parser.add_argument("--output-dataset-dir", required=True, type=Path)
    parser.add_argument("--target-col", required=True, type=int)
    parser.add_argument("--missing-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_tabular_benchmark(
        input_matrix_path=args.input_matrix,
        output_dataset_dir=args.output_dataset_dir,
        target_col=args.target_col,
        missing_fraction=args.missing_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    print(f"wrote tabular benchmark: {args.output_dataset_dir}")


if __name__ == "__main__":
    main()

