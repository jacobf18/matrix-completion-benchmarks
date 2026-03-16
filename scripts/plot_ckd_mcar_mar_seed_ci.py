#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def _pretty_text(raw: str) -> str:
    text = str(raw).replace("_", " ").strip()
    words = [w.capitalize() for w in text.split()]
    return " ".join(words)


def _pretty_method(method: str) -> str:
    if method == "tab_impute":
        return "TabImpute"
    if method == "tab_impute_constraints":
        return "TabImputeConstraints"
    if method == "TabImputeConstraints":
        return "TabImputeConstraints"
    return _pretty_text(method)


def _parse_csv_list(raw: str) -> list[str]:
    vals = [token.strip() for token in raw.split(",") if token.strip()]
    if not vals:
        raise ValueError("Expected at least one comma-separated value.")
    return vals


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot CKD metrics for MCAR and MAR patterns as line plots with 95% confidence intervals "
            "across seeds (x-axis: missingness fraction)."
        )
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("benchmarks/reports/ckd_ehr_all_algorithms/ckd_ehr_classification_summary.csv"),
        help="Classification summary CSV path.",
    )
    parser.add_argument(
        "--patterns",
        type=str,
        default="mcar,mar_logistic",
        help="Comma-separated patterns to include.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="",
        help="Optional comma-separated metric columns. Default: all numeric metrics.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="",
        help="Optional comma-separated methods to include. Default: all methods.",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=3,
        help="Number of subplot columns.",
    )
    parser.add_argument(
        "--figscale",
        type=float,
        default=4.0,
        help="Per-subplot figure scale in inches.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("benchmarks/reports/ckd_ehr_all_algorithms/ckd_mcar_mar_metrics_seed_ci.png"),
        help="Output plot PNG path.",
    )
    return parser


def _is_number(raw: str) -> bool:
    try:
        float(raw)
        return True
    except Exception:
        return False


def _is_metric_col(col: str) -> bool:
    non_metrics = {"bundle", "method", "missing_fraction", "pattern", "seed", "status", "error"}
    return col not in non_metrics


def main() -> None:
    args = build_parser().parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_ckd_mcar_mar_seed_ci.py requires matplotlib. Install with: pip install matplotlib") from exc
    plt.rcParams["font.family"] = "serif"

    if not args.results_csv.exists():
        raise FileNotFoundError(f"Results CSV not found: {args.results_csv}")

    patterns = set(_parse_csv_list(args.patterns))
    methods_filter = set(_parse_csv_list(args.methods)) if args.methods.strip() else None

    with args.results_csv.open("r", newline="", encoding="utf-8") as fh:
        rows = [row for row in csv.DictReader(fh)]
    if not rows:
        raise ValueError(f"No rows found in: {args.results_csv}")

    filtered_rows = []
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        pattern = str(row.get("pattern", ""))
        if pattern not in patterns:
            continue
        if methods_filter is not None and str(row.get("method", "")) not in methods_filter:
            continue
        frac = str(row.get("missing_fraction", "")).strip()
        if not _is_number(frac):
            continue
        filtered_rows.append(row)

    if not filtered_rows:
        raise ValueError("No rows left after filtering by patterns/methods/status and finite missing_fraction.")

    if args.metrics.strip():
        metrics = _parse_csv_list(args.metrics)
    else:
        candidate_metrics = [c for c in filtered_rows[0].keys() if _is_metric_col(c)]
        metrics = []
        for c in candidate_metrics:
            if all((str(r.get(c, "")).strip() == "" or _is_number(str(r.get(c, "")))) for r in filtered_rows):
                metrics.append(c)
    if not metrics:
        raise ValueError("No numeric metric columns selected.")

    methods = sorted({str(r.get("method", "")) for r in filtered_rows})
    pattern_order = sorted(patterns)

    # metric -> pattern -> method -> fraction -> [values across seeds]
    grouped: dict[str, dict[str, dict[str, dict[float, list[float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    for row in filtered_rows:
        pattern = str(row["pattern"])
        method = str(row["method"])
        frac = float(row["missing_fraction"])
        for metric in metrics:
            raw = str(row.get(metric, "")).strip()
            if raw == "":
                continue
            if not _is_number(raw):
                continue
            grouped[metric][pattern][method][frac].append(float(raw))

    n_metrics = len(metrics)
    ncols = max(1, args.ncols)
    nrows = math.ceil(n_metrics / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(args.figscale * ncols, args.figscale * nrows), squeeze=False)

    linestyles = ["-", "--", "-.", ":"]
    style_map = {p: linestyles[i % len(linestyles)] for i, p in enumerate(pattern_order)}
    colors = plt.get_cmap("tab10", max(1, len(methods)))

    for idx, metric in enumerate(metrics):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        for m_idx, method in enumerate(methods):
            color = colors(m_idx)
            for pattern in pattern_order:
                frac_to_vals = grouped[metric][pattern][method]
                if not frac_to_vals:
                    continue
                fracs = sorted(frac_to_vals.keys())
                means = np.array([float(np.mean(frac_to_vals[f])) for f in fracs], dtype=np.float64)
                cis = []
                for f in fracs:
                    vals = np.asarray(frac_to_vals[f], dtype=np.float64)
                    if vals.size <= 1:
                        cis.append(0.0)
                    else:
                        cis.append(float(1.96 * np.std(vals, ddof=1) / np.sqrt(vals.size)))
                ci_arr = np.asarray(cis, dtype=np.float64)
                label = _pretty_method(method)
                ax.plot(fracs, means, linestyle=style_map[pattern], color=color, linewidth=1.8, label=label)
                ax.fill_between(fracs, means - ci_arr, means + ci_arr, color=color, alpha=0.15)

        ax.set_title(_pretty_text(metric), fontsize=10)
        ax.set_xlabel("Missingness Fraction")
        ax.set_ylabel(_pretty_text(metric))
        ax.grid(alpha=0.25)

    for idx in range(n_metrics, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    handles, labels = [], []
    for ax_row in axes:
        for ax in ax_row:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
    if handles:
        dedup = {}
        for h, l in zip(handles, labels):
            if l not in dedup:
                dedup[l] = h
        fig.legend(
            dedup.values(),
            dedup.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(7, len(dedup)),
            fontsize=8,
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.88))
    plt.savefig(args.output_path, dpi=180)
    print(f"wrote plot: {args.output_path}")


if __name__ == "__main__":
    main()
