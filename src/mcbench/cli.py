from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from .algorithms import ALGORITHM_REGISTRY
from .datasets.catalog import load_catalog
from .datasets.downloader import build_downloader, write_download_manifest
from .datasets.noisy import make_noisy_matrix
from .metrics import METRIC_REGISTRY
from .workflows.evaluate import evaluate_prediction
from .workflows.prepare import prepare_random_holdout
from .workflows.run_algorithm import run_algorithm


def _json_dict(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return value


def _load_plugins(modules: list[str]) -> None:
    for module in modules:
        importlib.import_module(module)


def _default_catalog_path() -> Path:
    return Path(__file__).resolve().parents[2] / "benchmarks" / "catalog.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcbench",
        description="Benchmarks for matrix completion algorithms.",
    )
    parser.add_argument(
        "--plugin",
        action="append",
        default=[],
        help="Import a Python module that registers algorithms/metrics.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare-split", help="Create observed/eval split from a complete matrix.")
    prepare_parser.add_argument("--input-matrix", required=True, type=Path)
    prepare_parser.add_argument("--output-dataset-dir", required=True, type=Path)
    prepare_parser.add_argument("--holdout-fraction", required=True, type=float)
    prepare_parser.add_argument("--seed", type=int, default=0)

    run_parser = subparsers.add_parser("run-algorithm", help="Run one algorithm and save predictions.")
    run_parser.add_argument("--dataset-dir", required=True, type=Path)
    run_parser.add_argument("--algorithm", required=True)
    run_parser.add_argument("--output-dir", required=True, type=Path)
    run_parser.add_argument("--params-json", default="{}")

    eval_parser = subparsers.add_parser("evaluate", help="Score saved prediction file.")
    eval_parser.add_argument("--dataset-dir", required=True, type=Path)
    eval_parser.add_argument("--prediction-path", required=True, type=Path)
    eval_parser.add_argument("--metrics", nargs="+", required=True)
    eval_parser.add_argument("--output-path", required=True, type=Path)
    eval_parser.add_argument("--metric-params-json", default="{}")

    subparsers.add_parser("list-algorithms", help="List registered algorithms.")
    subparsers.add_parser("list-metrics", help="List registered metrics.")

    ds_list = subparsers.add_parser("list-datasets", help="List datasets and noisy benchmark presets from catalog.")
    ds_list.add_argument("--catalog-path", type=Path, default=_default_catalog_path())

    fetch = subparsers.add_parser("fetch-dataset", help="Download one or more source datasets from catalog.")
    fetch.add_argument("--dataset-id", nargs="+", required=True)
    fetch.add_argument("--catalog-path", type=Path, default=_default_catalog_path())
    fetch.add_argument("--output-root", type=Path, default=Path("benchmarks/sources"))
    fetch.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip datasets that already have processed outputs in output-root.",
    )
    fetch.add_argument(
        "--force",
        action="store_true",
        help="Force redownload/rebuild even if dataset already exists.",
    )

    noisy = subparsers.add_parser("make-noisy", help="Generate noisy matrix benchmark from catalog preset.")
    noisy.add_argument("--benchmark-id", required=True)
    noisy.add_argument("--catalog-path", type=Path, default=_default_catalog_path())
    noisy.add_argument("--source-root", type=Path, default=Path("benchmarks/sources"))
    noisy.add_argument("--output-root", type=Path, default=Path("benchmarks/noisy_sources"))
    noisy.add_argument("--seed", type=int, default=0)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _load_plugins(args.plugin)

    if args.command == "prepare-split":
        prepare_random_holdout(
            input_matrix_path=args.input_matrix,
            output_dataset_dir=args.output_dataset_dir,
            holdout_fraction=args.holdout_fraction,
            seed=args.seed,
        )
        return

    if args.command == "run-algorithm":
        run_algorithm(
            dataset_dir=args.dataset_dir,
            algorithm_name=args.algorithm,
            output_dir=args.output_dir,
            algorithm_params=_json_dict(args.params_json),
        )
        return

    if args.command == "evaluate":
        metric_params = _json_dict(args.metric_params_json)
        evaluate_prediction(
            dataset_dir=args.dataset_dir,
            prediction_path=args.prediction_path,
            metric_names=args.metrics,
            output_path=args.output_path,
            metric_params=metric_params,
        )
        return

    if args.command == "list-algorithms":
        for name in sorted(ALGORITHM_REGISTRY):
            print(name)
        return

    if args.command == "list-metrics":
        for name in sorted(METRIC_REGISTRY):
            print(name)
        return

    if args.command == "list-datasets":
        catalog = load_catalog(args.catalog_path)
        print("datasets:")
        for dataset_id, spec in sorted(catalog.datasets.items()):
            print(f"  - {dataset_id}: {spec.description}")
        print("noisy_benchmarks:")
        for bench_id, spec in sorted(catalog.noisy_benchmarks.items()):
            print(f"  - {bench_id} (base={spec.base_dataset}, noise_type={spec.noise_type})")
        return

    if args.command == "fetch-dataset":
        catalog = load_catalog(args.catalog_path)
        output_root = args.output_root
        if args.skip_existing and args.force:
            raise ValueError("--skip-existing and --force cannot be used together.")
        downloaded_dirs: list[Path] = []
        for dataset_id in args.dataset_id:
            spec = catalog.datasets.get(dataset_id)
            if spec is None:
                known = ", ".join(sorted(catalog.datasets))
                raise ValueError(f"Unknown dataset_id '{dataset_id}'. Known: {known}")
            downloader = build_downloader(spec)
            if args.skip_existing and downloader.is_ready(output_root=output_root):
                output_dir = output_root / dataset_id
                downloaded_dirs.append(output_dir)
                print(f"skipped (already exists): {dataset_id} -> {output_dir}")
                continue
            try:
                output_dir = downloader.fetch(output_root=output_root)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed fetching dataset '{dataset_id}' from '{spec.source_url}'."
                ) from exc
            downloaded_dirs.append(output_dir)
            print(f"downloaded: {dataset_id} -> {output_dir}")
        write_download_manifest(output_root=output_root, dataset_dirs=downloaded_dirs)
        return

    if args.command == "make-noisy":
        catalog = load_catalog(args.catalog_path)
        spec = catalog.noisy_benchmarks.get(args.benchmark_id)
        if spec is None:
            known = ", ".join(sorted(catalog.noisy_benchmarks))
            raise ValueError(f"Unknown benchmark_id '{args.benchmark_id}'. Known: {known}")
        input_matrix_path = args.source_root / spec.base_dataset / "matrix.npy"
        if not input_matrix_path.exists():
            raise FileNotFoundError(
                f"Base matrix not found at {input_matrix_path}. Run fetch-dataset first."
            )
        out_dir = args.output_root / args.benchmark_id
        out_matrix_path = out_dir / "matrix.npy"
        make_noisy_matrix(
            input_matrix_path=input_matrix_path,
            output_matrix_path=out_matrix_path,
            noise_type=spec.noise_type,
            params=spec.params,
            seed=args.seed,
        )
        print(f"generated noisy benchmark: {args.benchmark_id} -> {out_matrix_path}")
        return

    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
