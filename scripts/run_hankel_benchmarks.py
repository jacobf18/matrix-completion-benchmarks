#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
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


def _save_dataset_snapshot(output_dir: Path, preset_id: str, benchmark: Any, preset: Any) -> Path:
    snapshot_dir = output_dir / "repro" / preset_id / "dataset"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    np.save(snapshot_dir / "clean_signal.npy", benchmark.clean_signal)
    np.save(snapshot_dir / "observed_signal.npy", benchmark.observed_signal)
    np.save(snapshot_dir / "missing_mask.npy", benchmark.missing_mask.astype(bool))
    np.save(snapshot_dir / "hankel_observed.npy", benchmark.hankel_observed)
    save_json(
        snapshot_dir / "dataset_meta.json",
        {
            "preset_id": preset_id,
            "params": preset.params,
            "signal_model": preset.signal_model,
            "mask_type": preset.mask_type,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return snapshot_dir


def _save_algorithm_artifacts(
    output_dir: Path,
    preset_id: str,
    algorithm: str,
    reconstructed: np.ndarray | None,
    missing_mask: np.ndarray,
    clean_signal: np.ndarray,
    status: str,
    error: str,
    algo_params: dict[str, Any],
) -> None:
    algo_dir = output_dir / "repro" / preset_id / "algorithms" / algorithm
    algo_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "preset_id": preset_id,
        "algorithm": algorithm,
        "status": status,
        "error": error,
        "algorithm_params": algo_params,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if reconstructed is not None:
        np.save(algo_dir / "reconstructed_signal.npy", reconstructed)
        miss = missing_mask.astype(bool)
        np.save(algo_dir / "eval_true_values.npy", clean_signal[miss])
        np.save(algo_dir / "eval_pred_values.npy", reconstructed[miss])
        payload["eval_count"] = int(np.sum(miss))
    save_json(algo_dir / "artifacts_meta.json", payload)


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
        _save_dataset_snapshot(output_dir=args.output_dir, preset_id=preset_id, benchmark=benchmark, preset=preset)
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

            try:
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
                _save_algorithm_artifacts(
                    output_dir=args.output_dir,
                    preset_id=preset_id,
                    algorithm=algo,
                    reconstructed=reconstructed,
                    missing_mask=benchmark.missing_mask,
                    clean_signal=benchmark.clean_signal,
                    status="ok",
                    error="",
                    algo_params=algo_params,
                )

                rows.append(
                    {
                        "preset_id": preset_id,
                        "algorithm": algo,
                        "noise_sigma": sigma,
                        "missing_fraction": miss,
                        "signal_model": preset.signal_model,
                        "mask_type": preset.mask_type,
                        "status": "ok",
                        "error": "",
                        "rmse_all": metrics["rmse_all"],
                        "rmse_missing": metrics["rmse_missing"],
                        "nrmse_missing": metrics["nrmse_missing"],
                    }
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                _save_algorithm_artifacts(
                    output_dir=args.output_dir,
                    preset_id=preset_id,
                    algorithm=algo,
                    reconstructed=None,
                    missing_mask=benchmark.missing_mask,
                    clean_signal=benchmark.clean_signal,
                    status="failed",
                    error=error,
                    algo_params=algo_params,
                )
                rows.append(
                    {
                        "preset_id": preset_id,
                        "algorithm": algo,
                        "noise_sigma": sigma,
                        "missing_fraction": miss,
                        "signal_model": preset.signal_model,
                        "mask_type": preset.mask_type,
                        "status": "failed",
                        "error": error,
                        "rmse_all": np.nan,
                        "rmse_missing": np.nan,
                        "nrmse_missing": np.nan,
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
                "status",
                "error",
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
