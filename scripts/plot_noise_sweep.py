#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot NRMSE vs noise level from noise sweep CSV results.")
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("benchmarks/reports/noise_sweep/noise_sweep_results.csv"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--metric",
        choices=["rmse", "nrmse", "mae", "nmae"],
        default="nrmse",
        help="Metric column to plot on the y-axis.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_noise_sweep.py requires matplotlib. Install with: pip install matplotlib") from exc

    if not args.results_csv.exists():
        raise FileNotFoundError(f"Results CSV not found: {args.results_csv}")

    if args.output_path is None:
        args.output_path = Path(f"benchmarks/reports/noise_sweep/{args.metric}_vs_noise.png")

    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with args.results_csv.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            algorithm = str(row["algorithm"])
            sigma = float(row["noise_sigma"])
            metric_value = float(row[args.metric])
            series[algorithm].append((sigma, metric_value))

    if not series:
        raise ValueError(f"No rows found in {args.results_csv}")

    plt.figure(figsize=(8, 5))
    for algorithm, points in sorted(series.items()):
        points_sorted = sorted(points, key=lambda x: x[0])
        xs = [p[0] for p in points_sorted]
        ys = [p[1] for p in points_sorted]
        plt.plot(xs, ys, marker="o", linewidth=2, label=algorithm)

    metric_label = {
        "rmse": "RMSE",
        "nrmse": "Normalized RMSE",
        "mae": "MAE",
        "nmae": "Normalized MAE",
    }[args.metric]
    plt.xlabel("Noise level (sigma)")
    plt.ylabel(metric_label)
    plt.title(f"{metric_label} vs Noise Level")
    plt.grid(alpha=0.3)
    plt.legend()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output_path, dpi=150)
    print(f"wrote plot: {args.output_path}")


if __name__ == "__main__":
    main()
