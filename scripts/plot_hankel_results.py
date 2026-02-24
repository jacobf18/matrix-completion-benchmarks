#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Hankel benchmark results.")
    parser.add_argument("--results-csv", type=Path, default=Path("benchmarks/reports/hankel/hankel_results.csv"))
    parser.add_argument("--metric", choices=["rmse_all", "rmse_missing", "nrmse_missing"], default="nrmse_missing")
    parser.add_argument("--x-axis", choices=["noise_sigma", "missing_fraction"], default="noise_sigma")
    parser.add_argument("--output-path", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output_path is None:
        args.output_path = Path(f"benchmarks/reports/hankel/{args.metric}_vs_{args.x_axis}.png")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_hankel_results.py requires matplotlib (pip install matplotlib)") from exc

    rows = []
    with args.results_csv.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            row[args.x_axis] = float(row[args.x_axis])
            row[args.metric] = float(row[args.metric])
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows in {args.results_csv}")

    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        grouped[row["algorithm"]].append((row[args.x_axis], row[args.metric]))

    plt.figure(figsize=(8, 5))
    for algo, pts in sorted(grouped.items()):
        pts = sorted(pts, key=lambda p: p[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", linewidth=2, label=algo)

    plt.xlabel("Noise sigma" if args.x_axis == "noise_sigma" else "Missing fraction")
    plt.ylabel(args.metric.replace("_", " ").upper())
    plt.title(f"Hankel Completion: {args.metric} vs {args.x_axis}")
    plt.grid(alpha=0.3)
    plt.legend()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output_path, dpi=150)
    print(f"wrote plot: {args.output_path}")


if __name__ == "__main__":
    main()

