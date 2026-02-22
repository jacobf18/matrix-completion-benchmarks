from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    kind: str
    description: str
    source_url: str
    mirror_urls: tuple[str, ...]
    citation_url: str


@dataclass(frozen=True)
class NoisyBenchmarkSpec:
    benchmark_id: str
    base_dataset: str
    description: str
    noise_type: str
    citation_url: str | None
    params: dict[str, Any]


@dataclass(frozen=True)
class SyntheticNoisyBenchmarkSpec:
    benchmark_id: str
    description: str
    dgp_type: str
    noise_type: str
    params: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkCatalog:
    version: int
    datasets: dict[str, DatasetSpec]
    noisy_benchmarks: dict[str, NoisyBenchmarkSpec]
    synthetic_noisy_benchmarks: dict[str, SyntheticNoisyBenchmarkSpec]


def load_catalog(path: Path) -> BenchmarkCatalog:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Invalid catalog structure.")

    raw_datasets = payload.get("datasets", {})
    raw_noisy = payload.get("noisy_benchmarks", {})
    raw_synth = payload.get("synthetic_noisy_benchmarks", {})
    if not isinstance(raw_datasets, dict) or not isinstance(raw_noisy, dict):
        raise ValueError("Catalog must contain mapping objects for datasets and noisy_benchmarks.")
    if not isinstance(raw_synth, dict):
        raise ValueError("Catalog synthetic_noisy_benchmarks must be a mapping.")

    datasets: dict[str, DatasetSpec] = {}
    for dataset_id, value in raw_datasets.items():
        if not isinstance(value, dict):
            raise ValueError(f"Dataset entry '{dataset_id}' must be a mapping.")
        datasets[dataset_id] = DatasetSpec(
            dataset_id=dataset_id,
            kind=str(value["kind"]),
            description=str(value["description"]),
            source_url=str(value["source_url"]),
            mirror_urls=tuple(str(url) for url in value.get("mirror_urls", [])),
            citation_url=str(value["citation_url"]),
        )

    noisy_benchmarks: dict[str, NoisyBenchmarkSpec] = {}
    for benchmark_id, value in raw_noisy.items():
        if not isinstance(value, dict):
            raise ValueError(f"Noisy benchmark entry '{benchmark_id}' must be a mapping.")
        noisy_benchmarks[benchmark_id] = NoisyBenchmarkSpec(
            benchmark_id=benchmark_id,
            base_dataset=str(value["base_dataset"]),
            description=str(value["description"]),
            noise_type=str(value["noise_type"]),
            citation_url=str(value["citation_url"]) if value.get("citation_url") else None,
            params=dict(value.get("params", {})),
        )

    synthetic_noisy_benchmarks: dict[str, SyntheticNoisyBenchmarkSpec] = {}
    for benchmark_id, value in raw_synth.items():
        if not isinstance(value, dict):
            raise ValueError(f"Synthetic benchmark entry '{benchmark_id}' must be a mapping.")
        synthetic_noisy_benchmarks[benchmark_id] = SyntheticNoisyBenchmarkSpec(
            benchmark_id=benchmark_id,
            description=str(value["description"]),
            dgp_type=str(value["dgp_type"]),
            noise_type=str(value["noise_type"]),
            params=dict(value.get("params", {})),
        )

    return BenchmarkCatalog(
        version=int(payload.get("version", 1)),
        datasets=datasets,
        noisy_benchmarks=noisy_benchmarks,
        synthetic_noisy_benchmarks=synthetic_noisy_benchmarks,
    )
