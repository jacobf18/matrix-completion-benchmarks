#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from mcbench.algorithms import ALGORITHM_REGISTRY
from mcbench.io import load_matrix, save_json


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run TabImputeV2 standalone on saved synthetic denoising benchmark matrices."
    )
    parser.add_argument(
        "--repro-root",
        type=Path,
        default=Path("benchmarks/reports/synthetic_denoise_no_forestdiffusion/repro"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/reports/tabimpute_v2_standalone"),
    )
    parser.add_argument("--preset-ids", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-num-rows", type=int, default=None)
    parser.add_argument("--max-num-chunks", type=int, default=None)
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    algo_cls = ALGORITHM_REGISTRY.get("tab_impute")
    if algo_cls is None:
        raise ValueError("Algorithm 'tab_impute' is not registered.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    preset_dirs = sorted(p for p in args.repro_root.iterdir() if p.is_dir())
    if args.preset_ids is not None:
        wanted = set(args.preset_ids)
        preset_dirs = [p for p in preset_dirs if p.name in wanted]

    seed_filter = None if args.seeds is None else {f"seed_{seed}" for seed in args.seeds}

    for preset_dir in preset_dirs:
        for seed_dir in sorted(p for p in preset_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")):
            if seed_filter is not None and seed_dir.name not in seed_filter:
                continue

            dataset_dir = seed_dir / "dataset"
            observed = load_matrix(dataset_dir / "observed.npy")
            y_true = load_matrix(dataset_dir / "ground_truth.npy")
            eval_mask = np.load(dataset_dir / "eval_mask.npy").astype(bool)
            dataset_meta = json.loads((dataset_dir / "dataset_meta.json").read_text())
            params = {
                "device": args.device,
                "model_version": 2,
                "v2_checkpoint_path": args.checkpoint_path,
                "max_num_rows": args.max_num_rows,
                "max_num_chunks": args.max_num_chunks,
                "num_repeats": args.num_repeats,
                "verbose": args.verbose,
            }

            try:
                pred = np.asarray(algo_cls().complete(observed, **params), dtype=np.float64)
                rmse, nrmse = _rmse_and_nrmse(y_true=y_true, y_pred=pred, eval_mask=eval_mask)
                status = "ok"
                error = ""
            except Exception as exc:
                pred = None
                rmse = np.nan
                nrmse = np.nan
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"

            row = {
                "preset_id": preset_dir.name,
                "seed": seed_dir.name.removeprefix("seed_"),
                "noise_sigma": dataset_meta["params"].get("sigma", np.nan),
                "noise_type": dataset_meta.get("noise_type", ""),
                "status": status,
                "error": error,
                "rmse": rmse,
                "nrmse": nrmse,
                "eval_count": int(np.sum(eval_mask)),
            }
            rows.append(row)

            pred_dir = args.output_dir / "predictions" / preset_dir.name / seed_dir.name
            pred_dir.mkdir(parents=True, exist_ok=True)
            save_json(pred_dir / "meta.json", {**row, "params": params})
            if pred is not None:
                np.save(pred_dir / "prediction_full.npy", pred)

    detailed_path = args.output_dir / "tabimpute_v2_detailed.csv"
    with detailed_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["preset_id", "seed", "noise_sigma", "noise_type", "status", "error", "rmse", "nrmse", "eval_count"],
        )
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        key = (str(row["preset_id"]), float(row["noise_sigma"]), str(row["noise_type"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (preset_id, noise_sigma, noise_type), group_rows in sorted(grouped.items(), key=lambda item: item[0][1]):
        rmse_vals = np.array([float(r["rmse"]) for r in group_rows], dtype=np.float64)
        nrmse_vals = np.array([float(r["nrmse"]) for r in group_rows], dtype=np.float64)
        summary_rows.append(
            {
                "preset_id": preset_id,
                "algorithm": "tab_impute_v2",
                "noise_sigma": noise_sigma,
                "noise_type": noise_type,
                "num_runs": len(group_rows),
                "rmse": float(np.mean(rmse_vals)),
                "rmse_std": float(np.std(rmse_vals)),
                "nrmse": float(np.mean(nrmse_vals)),
                "nrmse_std": float(np.std(nrmse_vals)),
            }
        )

    summary_path = args.output_dir / "tabimpute_v2_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["preset_id", "algorithm", "noise_sigma", "noise_type", "num_runs", "rmse", "rmse_std", "nrmse", "nrmse_std"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    save_json(
        args.output_dir / "tabimpute_v2_summary.json",
        {
            "repro_root": str(args.repro_root),
            "checkpoint_path": str(args.checkpoint_path),
            "device": args.device,
            "rows": summary_rows,
        },
    )

    print(f"wrote: {detailed_path}")
    print(f"wrote: {summary_path}")


if __name__ == "__main__":
    main()
