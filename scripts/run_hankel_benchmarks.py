#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from mcbench.datasets.catalog import load_catalog
from mcbench.io import save_json
from mcbench.workflows.hankel import (
    evaluate_reconstruction,
    generate_hankel_benchmark,
    reconstruct_signal_with_method,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Hankel matrix-completion benchmark presets.")
    parser.add_argument("--catalog-path", type=Path, default=Path("benchmarks/catalog.yaml"))
    parser.add_argument(
        "--preset-ids",
        nargs="+",
        default=["hankel_gaussian_sigma_0p01", "hankel_gaussian_sigma_0p05", "hankel_gaussian_sigma_0p10"],
    )
    parser.add_argument("--algorithms", nargs="+", default=["global_mean", "soft_impute", "cadzow"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/reports/hankel"))
    parser.add_argument("--save-signals", action="store_true")
    parser.add_argument(
        "--soft-impute-params-json",
        default='{"max_iters": 150, "tol": 1e-5, "init_fill": "zero"}',
    )
    parser.add_argument("--cadzow-params-json", default='{"max_iters": 120, "tol": 1e-6}')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    soft_params: dict[str, Any] = json.loads(args.soft_impute_params_json)
    cadzow_params: dict[str, Any] = json.loads(args.cadzow_params_json)
    catalog = load_catalog(args.catalog_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for preset_id in args.preset_ids:
        preset = catalog.hankel_benchmarks.get(preset_id)
        if preset is None:
            known = ", ".join(sorted(catalog.hankel_benchmarks))
            raise ValueError(f"Unknown hankel preset '{preset_id}'. Known: {known}")

        benchmark = generate_hankel_benchmark(
            signal_model=preset.signal_model,
            mask_type=preset.mask_type,
            params=preset.params,
            seed=args.seed,
        )
        rank = int(preset.params.get("rank", 8))
        sigma = float(preset.params.get("noise_sigma", np.nan))
        miss = float(preset.params.get("missing_fraction", np.nan))

        if args.save_signals:
            dataset_dir = args.output_dir / "signals" / preset_id
            dataset_dir.mkdir(parents=True, exist_ok=True)
            np.save(dataset_dir / "clean_signal.npy", benchmark.clean_signal)
            np.save(dataset_dir / "observed_signal.npy", benchmark.observed_signal)
            np.save(dataset_dir / "missing_mask.npy", benchmark.missing_mask)
            save_json(dataset_dir / "preset_meta.json", {"preset_id": preset_id, "params": preset.params})

        for algo in args.algorithms:
            algo_params: dict[str, Any] = {}
            if algo == "soft_impute":
                algo_params = dict(soft_params)
            elif algo == "cadzow":
                algo_params = dict(cadzow_params)

            reconstructed = reconstruct_signal_with_method(
                hankel_observed=benchmark.hankel_observed,
                algorithm_name=algo,
                algorithm_params=algo_params,
                hankel_rank=rank,
            )
            metrics = evaluate_reconstruction(
                clean_signal=benchmark.clean_signal,
                reconstructed_signal=reconstructed,
                missing_mask=benchmark.missing_mask,
            )

            if args.save_signals:
                np.save(args.output_dir / "signals" / preset_id / f"{algo}_reconstructed.npy", reconstructed)

            rows.append(
                {
                    "preset_id": preset_id,
                    "algorithm": algo,
                    "noise_sigma": sigma,
                    "missing_fraction": miss,
                    "signal_model": preset.signal_model,
                    "mask_type": preset.mask_type,
                    "rmse_all": metrics["rmse_all"],
                    "rmse_missing": metrics["rmse_missing"],
                    "nrmse_missing": metrics["nrmse_missing"],
                }
            )

    csv_path = args.output_dir / "hankel_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "preset_id",
                "algorithm",
                "noise_sigma",
                "missing_fraction",
                "signal_model",
                "mask_type",
                "rmse_all",
                "rmse_missing",
                "nrmse_missing",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote results: {csv_path}")


if __name__ == "__main__":
    main()

