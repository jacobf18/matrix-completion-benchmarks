from __future__ import annotations

import csv
import io
import json
import re
import shutil
import subprocess
import urllib.error
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ..io import save_json, save_matrix
from .catalog import DatasetSpec


def _download_url(urls: list[str], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in urls:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "mcbench/0.1"})
            with urllib.request.urlopen(request) as response:
                destination.write_bytes(response.read())
            return destination
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            last_error = exc
    failed = ", ".join(urls)
    raise RuntimeError(f"Failed to download from all candidate URLs: {failed}") from last_error


def _dense_from_triplets(
    row_ids: np.ndarray,
    col_ids: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_rows = np.unique(row_ids)
    unique_cols = np.unique(col_ids)
    row_to_idx = {int(v): i for i, v in enumerate(unique_rows)}
    col_to_idx = {int(v): i for i, v in enumerate(unique_cols)}

    matrix = np.full((unique_rows.size, unique_cols.size), np.nan, dtype=np.float64)
    for r, c, val in zip(row_ids, col_ids, values):
        matrix[row_to_idx[int(r)], col_to_idx[int(c)]] = float(val)
    return matrix, unique_rows, unique_cols


def _find_column(columns: dict[str, str], candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    known = ", ".join(sorted(columns))
    wanted = ", ".join(candidates)
    raise ValueError(f"Could not find one of [{wanted}] in columns: {known}")


class DatasetDownloader(ABC):
    def __init__(self, spec: DatasetSpec) -> None:
        self.spec = spec

    @abstractmethod
    def fetch(self, output_root: Path) -> Path:
        """Download and materialize dataset matrix under output_root."""

    def _candidate_urls(self) -> list[str]:
        return [self.spec.source_url, *self.spec.mirror_urls]

    def _write_outputs(
        self,
        output_dir: Path,
        matrix: np.ndarray,
        row_labels: np.ndarray,
        col_labels: np.ndarray,
        raw_path: Path,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_matrix(output_dir / "matrix.npy", matrix)
        np.save(output_dir / "row_labels.npy", row_labels.astype(str))
        np.save(output_dir / "col_labels.npy", col_labels.astype(str))
        save_json(
            output_dir / "source_meta.json",
            {
                "dataset_id": self.spec.dataset_id,
                "kind": self.spec.kind,
                "description": self.spec.description,
                "source_url": self.spec.source_url,
                "citation_url": self.spec.citation_url,
                "raw_path": str(raw_path),
                "shape": list(matrix.shape),
                "known_count": int(np.sum(np.isfinite(matrix))),
            },
        )
        return output_dir

    def is_ready(self, output_root: Path) -> bool:
        dataset_dir = output_root / self.spec.dataset_id
        required = [
            dataset_dir / "matrix.npy",
            dataset_dir / "row_labels.npy",
            dataset_dir / "col_labels.npy",
            dataset_dir / "source_meta.json",
        ]
        return all(path.exists() for path in required)


class MovieLensDownloader(DatasetDownloader):
    def fetch(self, output_root: Path) -> Path:
        raw_dir = output_root / self.spec.dataset_id / "raw"
        raw_zip = _download_url(self._candidate_urls(), raw_dir / "archive.zip")
        with zipfile.ZipFile(raw_zip, "r") as zf:
            if self.spec.dataset_id == "movielens_100k":
                with zf.open("ml-100k/u.data", "r") as fh:
                    rows = [line.decode("utf-8").strip().split("\t") for line in fh if line.strip()]
                    row_ids = np.array([int(r[0]) for r in rows], dtype=np.int64)
                    col_ids = np.array([int(r[1]) for r in rows], dtype=np.int64)
                    values = np.array([float(r[2]) for r in rows], dtype=np.float64)
            elif self.spec.dataset_id == "movielens_1m":
                with zf.open("ml-1m/ratings.dat", "r") as fh:
                    rows = [line.decode("latin-1").strip().split("::") for line in fh if line.strip()]
                    row_ids = np.array([int(r[0]) for r in rows], dtype=np.int64)
                    col_ids = np.array([int(r[1]) for r in rows], dtype=np.int64)
                    values = np.array([float(r[2]) for r in rows], dtype=np.float64)
            elif self.spec.dataset_id == "movielens_latest_small":
                with zf.open("ml-latest-small/ratings.csv", "r") as fh:
                    reader = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8"))
                    row_ids, col_ids, values = [], [], []
                    for row in reader:
                        row_ids.append(int(row["userId"]))
                        col_ids.append(int(row["movieId"]))
                        values.append(float(row["rating"]))
                    row_ids = np.array(row_ids, dtype=np.int64)
                    col_ids = np.array(col_ids, dtype=np.int64)
                    values = np.array(values, dtype=np.float64)
            elif self.spec.dataset_id == "movielens_20m":
                with zf.open("ml-20m/ratings.csv", "r") as fh:
                    reader = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8"))
                    row_ids, col_ids, values = [], [], []
                    for row in reader:
                        row_ids.append(int(row["userId"]))
                        col_ids.append(int(row["movieId"]))
                        values.append(float(row["rating"]))
                    row_ids = np.array(row_ids, dtype=np.int64)
                    col_ids = np.array(col_ids, dtype=np.int64)
                    values = np.array(values, dtype=np.float64)
            else:
                raise ValueError(f"Unsupported MovieLens dataset: {self.spec.dataset_id}")

        matrix, users, items = _dense_from_triplets(row_ids=row_ids, col_ids=col_ids, values=values)
        return self._write_outputs(output_root / self.spec.dataset_id, matrix, users, items, raw_zip)


class RDatasetsPanelDownloader(DatasetDownloader):
    def fetch(self, output_root: Path) -> Path:
        raw_dir = output_root / self.spec.dataset_id / "raw"
        raw_csv = _download_url(self._candidate_urls(), raw_dir / "dataset.csv")

        rows: list[dict[str, str]] = []
        with raw_csv.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(row)

        if not rows:
            raise ValueError(f"Empty dataset for {self.spec.dataset_id}")

        colmap = {k.lower(): k for k in rows[0].keys()}
        unit_key = _find_column(colmap, ["state", "regionname", "region", "unit"])
        time_key = _find_column(colmap, ["year", "time", "period"])

        if self.spec.dataset_id == "prop99_smoking":
            value_key = _find_column(colmap, ["cigsale", "cig.sales", "outcome"])
        elif self.spec.dataset_id == "basque_gdpcap":
            value_key = _find_column(colmap, ["gdpcap", "gdp.cap", "outcome"])
        else:
            value_key = _find_column(colmap, ["outcome", "value"])

        units = sorted({row[unit_key] for row in rows})
        periods = sorted({int(float(row[time_key])) for row in rows})
        unit_to_idx = {u: i for i, u in enumerate(units)}
        period_to_idx = {t: i for i, t in enumerate(periods)}
        matrix = np.full((len(units), len(periods)), np.nan, dtype=np.float64)

        for row in rows:
            unit = row[unit_key]
            period = int(float(row[time_key]))
            try:
                value = float(row[value_key])
            except ValueError:
                continue
            matrix[unit_to_idx[unit], period_to_idx[period]] = value

        return self._write_outputs(
            output_root / self.spec.dataset_id,
            matrix,
            np.array(units, dtype=object),
            np.array(periods, dtype=np.int64),
            raw_csv,
        )


class MovieTweetingsDownloader(DatasetDownloader):
    def fetch(self, output_root: Path) -> Path:
        raw_dir = output_root / self.spec.dataset_id / "raw"
        raw_dat = _download_url(self._candidate_urls(), raw_dir / "ratings.dat")

        row_ids: list[int] = []
        col_ids: list[int] = []
        values: list[float] = []
        with raw_dat.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("::")
                if len(parts) < 3:
                    continue
                row_ids.append(int(parts[0]))
                col_ids.append(int(parts[1]))
                values.append(float(parts[2]))

        if not row_ids:
            raise ValueError("MovieTweetings ratings.dat is empty or malformed.")

        matrix, users, items = _dense_from_triplets(
            row_ids=np.array(row_ids, dtype=np.int64),
            col_ids=np.array(col_ids, dtype=np.int64),
            values=np.array(values, dtype=np.float64),
        )
        return self._write_outputs(output_root / self.spec.dataset_id, matrix, users, items, raw_dat)


class KaggleTabularDownloader(DatasetDownloader):
    def fetch(self, output_root: Path) -> Path:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "Kaggle tabular downloader requires pandas. Install with: pip install pandas"
            ) from exc

        raw_dir = output_root / self.spec.dataset_id / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        slug = self._extract_dataset_slug(self.spec.source_url)
        kaggle_bin = shutil.which("kaggle")
        if kaggle_bin is None:
            raise RuntimeError(
                "Kaggle CLI not found. Install/configure kaggle and authenticate (kaggle.json)."
            )

        cmd = [kaggle_bin, "datasets", "download", "-d", slug, "-p", str(raw_dir), "--force"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "Kaggle download failed. Ensure credentials are configured. "
                f"stderr: {proc.stderr.strip() or proc.stdout.strip()}"
            )

        zip_files = sorted(raw_dir.glob("*.zip"))
        if not zip_files:
            raise RuntimeError(f"No zip archive downloaded from Kaggle for slug '{slug}'.")
        archive_path = zip_files[0]
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(raw_dir)

        csv_candidates = sorted(raw_dir.rglob("*.csv"))
        if not csv_candidates:
            raise RuntimeError(f"No CSV files found after extracting {archive_path.name}.")
        csv_path = max(csv_candidates, key=lambda p: p.stat().st_size)

        df = pd.read_csv(csv_path)
        if df.shape[0] == 0 or df.shape[1] == 0:
            raise ValueError(f"CSV appears empty: {csv_path}")

        encoded = self._encode_tabular_dataframe(df)
        matrix = encoded.to_numpy(dtype=np.float64, copy=True)
        rows = np.arange(matrix.shape[0], dtype=np.int64)
        cols = np.array(encoded.columns.astype(str), dtype=object)
        output_dir = self._write_outputs(output_root / self.spec.dataset_id, matrix, rows, cols, archive_path)

        meta_path = output_dir / "source_meta.json"
        meta = json.loads(meta_path.read_text())
        meta["kaggle_dataset"] = slug
        meta["selected_csv"] = str(csv_path)
        save_json(meta_path, meta)
        return output_dir

    @staticmethod
    def _extract_dataset_slug(url: str) -> str:
        match = re.search(r"kaggle\.com/datasets/([^/]+/[^/]+)", url)
        if not match:
            raise ValueError(f"Could not parse Kaggle dataset slug from URL: {url}")
        return match.group(1)

    @staticmethod
    def _encode_tabular_dataframe(df: "pd.DataFrame") -> "pd.DataFrame":
        import pandas as pd

        out = pd.DataFrame(index=df.index)
        missing_tokens = {"", "na", "n/a", "null", "none", "?", "nan"}
        for col in df.columns:
            series = df[col]
            if series.dtype == object:
                normalized = series.astype(str).str.strip()
                normalized = normalized.mask(normalized.str.lower().isin(missing_tokens))
                series = normalized.replace("nan", np.nan)

            numeric = pd.to_numeric(series, errors="coerce")
            valid_numeric = int(numeric.notna().sum())
            valid_total = int(series.notna().sum()) if hasattr(series, "notna") else valid_numeric
            if valid_total > 0 and valid_numeric / valid_total >= 0.9:
                out[col] = numeric.astype(float)
                continue

            cats = pd.Categorical(series)
            codes = cats.codes.astype(float)
            codes[codes < 0] = np.nan
            out[col] = codes
        return out


def build_downloader(spec: DatasetSpec) -> DatasetDownloader:
    if spec.dataset_id.startswith("movielens_"):
        return MovieLensDownloader(spec)
    if spec.dataset_id == "movietweetings":
        return MovieTweetingsDownloader(spec)
    if spec.dataset_id in {"prop99_smoking", "basque_gdpcap"}:
        return RDatasetsPanelDownloader(spec)
    if spec.dataset_id in {"ckd_ehr_abu_dhabi"}:
        return KaggleTabularDownloader(spec)
    raise ValueError(f"No downloader implemented for '{spec.dataset_id}'.")


def write_download_manifest(output_root: Path, dataset_dirs: list[Path]) -> None:
    manifest = {
        "output_root": str(output_root),
        "dataset_dirs": [str(path) for path in dataset_dirs],
    }
    (output_root / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
