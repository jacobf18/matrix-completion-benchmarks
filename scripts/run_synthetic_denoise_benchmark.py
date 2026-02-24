#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from mcbench.algorithms import ALGORITHM_REGISTRY
from mcbench.datasets.catalog import load_catalog
from mcbench.datasets.simulated import generate_simulated_noisy_benchmark
from mcbench.io import load_matrix, save_json


DEFAULT_ALGORITHMS = [
    "global_mean",
    "row_mean",
    "soft_impute",
    "nuclear_norm_minimization",
    "hyperimpute",
    "missforest",
    "forest_diffusion",
]


def _parse_seeds(raw: str) -> list[int]:
    out = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not out:
        raise ValueError("At least one seed is required.")
    return out


def _rmse_and_nrmse(y_true: np.ndarray, y_pred: np.ndarray, eval_mask: np.ndarray) -> tuple[float, float]:
    valid = eval_mask & np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        raise ValueError("No valid eval cells for scoring.")
    t = y_true[valid]
    p = y_pred[valid]
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    denom = float(np.max(t) - np.min(t))
    nrmse = 0.0 if denom == 0 else rmse / denom
    return rmse, nrmse


def _noise_sigma_like(noise_type: str, params: dict[str, Any]) -> float:
    if noise_type == "gaussian":
        return float(params.get("sigma", np.nan))
    if noise_type == "student_t":
        return float(params.get("scale", np.nan))
    if noise_type == "sparse_corruption":
        return float(params.get("corruption_scale", np.nan))
    return float(np.nan)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End-to-end synthetic noisy matrix denoising benchmark across multiple algorithms."
    )
    parser.add_argument("--catalog-path", type=Path, default=Path("benchmarks/catalog.yaml"))
    parser.add_argument("--preset-ids", nargs="+", default=None, help="Defaults to all synthetic_noisy_benchmarks.")
    parser.add_argument("--algorithms", nargs="+", default=DEFAULT_ALGORITHMS)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--dataset-root", type=Path, default=Path("benchmarks/simulated/denoise_runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/reports/synthetic_denoise"))
    parser.add_argument("--algorithm-params-json", default="{}")
    parser.add_argument("--save-predictions", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    catalog = load_catalog(args.catalog_path)
    seeds = _parse_seeds(args.seeds)
    algorithm_params = json.loads(args.algorithm_params_json)
    if not isinstance(algorithm_params, dict):
        raise ValueError("algorithm-params-json must be a JSON object.")

    preset_ids = args.preset_ids or sorted(catalog.synthetic_noisy_benchmarks.keys())
    args.dataset_root.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detailed_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for preset_id in preset_ids:
        spec = catalog.synthetic_noisy_benchmarks.get(preset_id)
        if spec is None:
            known = ", ".join(sorted(catalog.synthetic_noisy_benchmarks))
            raise ValueError(f"Unknown synthetic preset '{preset_id}'. Known: {known}")

        for seed in seeds:
            dataset_dir = args.dataset_root / preset_id / f"seed_{seed}"
            generate_simulated_noisy_benchmark(
                output_dir=dataset_dir,
                dgp_type=spec.dgp_type,
                noise_type=spec.noise_type,
                params=spec.params,
                seed=seed,
            )
            observed = load_matrix(dataset_dir / "observed.npy")
            y_true = load_matrix(dataset_dir / "ground_truth.npy")
            eval_mask = np.load(dataset_dir / "eval_mask.npy").astype(bool)
            noise_sigma = _noise_sigma_like(spec.noise_type, spec.params)

            for algorithm_name in args.algorithms:
                algo_cls = ALGORITHM_REGISTRY.get(algorithm_name)
                if algo_cls is None:
                    detailed_rows.append(
                        {
                            "preset_id": preset_id,
                            "seed": seed,
                            "algorithm": algorithm_name,
                            "status": "failed",
                            "error": f"unknown algorithm '{algorithm_name}'",
                            "noise_sigma": noise_sigma,
                            "noise_type": spec.noise_type,
                            "rmse": "",
                            "nrmse": "",
                            "eval_count": int(np.sum(eval_mask)),
                        }
                    )
                    continue

                params = algorithm_params.get(algorithm_name, {})
                if not isinstance(params, dict):
                    raise ValueError(f"algorithm params for '{algorithm_name}' must be an object.")

                try:
                    pred = np.asarray(algo_cls().complete(observed, **params), dtype=np.float64)
                    rmse, nrmse = _rmse_and_nrmse(y_true=y_true, y_pred=pred, eval_mask=eval_mask)
                    error = ""
                    status = "ok"
                    if args.save_predictions:
                        pred_dir = args.output_dir / "predictions" / preset_id / f"seed_{seed}"
                        pred_dir.mkdir(parents=True, exist_ok=True)
                        np.save(pred_dir / f"{algorithm_name}.npy", pred)
                except Exception as exc:
                    rmse = np.nan
                    nrmse = np.nan
                    error = f"{type(exc).__name__}: {exc}"
                    status = "failed"

                detailed_rows.append(
                    {
                        "preset_id": preset_id,
                        "seed": seed,
                        "algorithm": algorithm_name,
                        "status": status,
                        "error": error,
                        "noise_sigma": noise_sigma,
                        "noise_type": spec.noise_type,
                        "rmse": rmse,
                        "nrmse": nrmse,
                        "eval_count": int(np.sum(eval_mask)),
                    }
                )

    # Aggregate successful rows for plotting/leaderboard.
    grouped: dict[tuple[str, str, float, str], list[dict[str, Any]]] = {}
    for row in detailed_rows:
        if row["status"] != "ok":
            continue
        key = (str(row["preset_id"]), str(row["algorithm"]), float(row["noise_sigma"]), str(row["noise_type"]))
        grouped.setdefault(key, []).append(row)

    for (preset_id, algorithm, noise_sigma, noise_type), rows in sorted(grouped.items()):
        rmse_vals = np.array([float(r["rmse"]) for r in rows], dtype=np.float64)
        nrmse_vals = np.array([float(r["nrmse"]) for r in rows], dtype=np.float64)
        summary_rows.append(
            {
                "preset_id": preset_id,
                "algorithm": algorithm,
                "noise_sigma": noise_sigma,
                "noise_type": noise_type,
                "num_runs": len(rows),
                "rmse": float(np.mean(rmse_vals)),
                "rmse_std": float(np.std(rmse_vals)),
                "nrmse": float(np.mean(nrmse_vals)),
                "nrmse_std": float(np.std(nrmse_vals)),
            }
        )

    detailed_csv = args.output_dir / "synthetic_denoise_detailed.csv"
    with detailed_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "preset_id",
                "seed",
                "algorithm",
                "status",
                "error",
                "noise_sigma",
                "noise_type",
                "rmse",
                "nrmse",
                "eval_count",
            ],
        )
        writer.writeheader()
        writer.writerows(detailed_rows)

    summary_csv = args.output_dir / "synthetic_denoise_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "preset_id",
                "algorithm",
                "noise_sigma",
                "noise_type",
                "num_runs",
                "rmse",
                "rmse_std",
                "nrmse",
                "nrmse_std",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    save_json(
        args.output_dir / "synthetic_denoise_summary.json",
        {
            "catalog_path": str(args.catalog_path),
            "preset_ids": preset_ids,
            "algorithms": args.algorithms,
            "seeds": seeds,
            "detailed_csv": str(detailed_csv),
            "summary_csv": str(summary_csv),
            "num_detailed_rows": len(detailed_rows),
            "num_summary_rows": len(summary_rows),
        },
    )

    failures = [r for r in detailed_rows if r["status"] != "ok"]
    print(f"wrote: {detailed_csv}")
    print(f"wrote: {summary_csv}")
    print(f"failures: {len(failures)}")


if __name__ == "__main__":
    main()

