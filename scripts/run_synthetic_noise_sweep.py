#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from mcbench.algorithms import ALGORITHM_REGISTRY
from mcbench.datasets.simulated import generate_simulated_noisy_benchmark
from mcbench.io import load_matrix


def _parse_noise_levels(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one noise level is required.")
    if any(v < 0 for v in values):
        raise ValueError("Noise levels must be non-negative.")
    return values


def _rmse_and_nrmse(y_true: np.ndarray, y_pred: np.ndarray, eval_mask: np.ndarray) -> tuple[float, float]:
    valid = eval_mask & np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        raise ValueError("No valid eval cells.")
    t = y_true[valid]
    p = y_pred[valid]
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    denom = float(np.max(t) - np.min(t))
    nrmse = 0.0 if denom == 0 else rmse / denom
    return rmse, nrmse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run synthetic noisy matrix sweep for selected algorithms.")
    parser.add_argument("--noise-levels", default="0.05,0.1,0.2,0.35,0.5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-root", type=Path, default=Path("benchmarks/simulated/noise_sweep"))
    parser.add_argument("--report-dir", type=Path, default=Path("benchmarks/reports/noise_sweep"))
    parser.add_argument("--dgp-type", default="low_rank_gaussian")
    parser.add_argument("--algorithms", nargs="+", default=["global_mean", "soft_impute", "forest_diffusion"])
    parser.add_argument(
        "--base-params-json",
        default='{"n_rows": 400, "n_cols": 300, "rank": 10, "observed_fraction": 0.35, "eval_fraction": 0.2, "factor_scale": 1.0}',
        help="JSON object of simulation parameters excluding sigma.",
    )
    parser.add_argument(
        "--soft-impute-params-json",
        default='{"max_iters": 150, "tol": 1e-5, "init_fill": "zero"}',
        help="JSON object passed only to soft_impute.",
    )
    parser.add_argument("--save-predictions", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    noise_levels = _parse_noise_levels(args.noise_levels)
    base_params = json.loads(args.base_params_json)
    soft_impute_params: dict[str, Any] = json.loads(args.soft_impute_params_json)

    args.dataset_root.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    results_path = args.report_dir / "noise_sweep_results.csv"
    rows: list[dict[str, Any]] = []

    for sigma in noise_levels:
        dataset_id = f"sigma_{sigma:.4f}".replace(".", "p")
        dataset_dir = args.dataset_root / dataset_id
        sim_params = dict(base_params)
        sim_params["sigma"] = sigma

        generate_simulated_noisy_benchmark(
            output_dir=dataset_dir,
            dgp_type=args.dgp_type,
            noise_type="gaussian",
            params=sim_params,
            seed=args.seed,
        )

        observed = load_matrix(dataset_dir / "observed.npy")
        y_true = load_matrix(dataset_dir / "ground_truth.npy")
        eval_mask = np.load(dataset_dir / "eval_mask.npy").astype(bool)

        for algorithm_name in args.algorithms:
            algo_cls = ALGORITHM_REGISTRY.get(algorithm_name)
            if algo_cls is None:
                known = ", ".join(sorted(ALGORITHM_REGISTRY))
                raise ValueError(f"Unknown algorithm '{algorithm_name}'. Known: {known}")

            algo_kwargs: dict[str, Any] = {}
            if algorithm_name == "soft_impute":
                algo_kwargs = dict(soft_impute_params)

            prediction = np.asarray(algo_cls().complete(observed, **algo_kwargs), dtype=np.float64)
            rmse, nrmse = _rmse_and_nrmse(y_true=y_true, y_pred=prediction, eval_mask=eval_mask)

            if args.save_predictions:
                pred_dir = args.report_dir / "predictions" / dataset_id
                pred_dir.mkdir(parents=True, exist_ok=True)
                np.save(pred_dir / f"{algorithm_name}.npy", prediction)

            rows.append(
                {
                    "dataset_id": dataset_id,
                    "noise_sigma": sigma,
                    "algorithm": algorithm_name,
                    "rmse": rmse,
                    "nrmse": nrmse,
                    "eval_count": int(np.sum(eval_mask)),
                }
            )

    with results_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["dataset_id", "noise_sigma", "algorithm", "rmse", "nrmse", "eval_count"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote results: {results_path}")


if __name__ == "__main__":
    main()

