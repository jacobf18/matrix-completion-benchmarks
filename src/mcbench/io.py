from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .types import DatasetBundle, Mask, Matrix


def load_matrix(path: Path) -> Matrix:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = np.load(path)
    elif suffix in {".csv", ".tsv"}:
        delimiter = "," if suffix == ".csv" else "\t"
        data = np.loadtxt(path, delimiter=delimiter)
    else:
        raise ValueError(f"Unsupported matrix format: {path}")
    return np.asarray(data, dtype=np.float64)


def save_matrix(path: Path, matrix: Matrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(matrix, dtype=np.float64))


def save_mask(path: Path, mask: Mask) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(mask, dtype=bool))


def load_bundle(dataset_dir: Path) -> DatasetBundle:
    return DatasetBundle(
        dataset_dir=dataset_dir,
        observed_path=dataset_dir / "observed.npy",
        ground_truth_path=dataset_dir / "ground_truth.npy",
        eval_mask_path=dataset_dir / "eval_mask.npy",
    )


def validate_bundle(bundle: DatasetBundle) -> None:
    missing = [
        path
        for path in [bundle.observed_path, bundle.ground_truth_path, bundle.eval_mask_path]
        if not path.exists()
    ]
    if missing:
        msg = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Dataset bundle is missing required files: {msg}")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

