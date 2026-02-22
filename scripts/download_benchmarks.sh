#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/download_benchmarks.sh

CATALOG_PATH="${CATALOG_PATH:-benchmarks/catalog.yaml}"
OUT_ROOT="${OUT_ROOT:-benchmarks/sources}"
NOISY_ROOT="${NOISY_ROOT:-benchmarks/noisy_sources}"

DATASETS=(
  "movielens_latest_small"
  "movielens_100k"
  "movielens_1m"
  "prop99_smoking"
  "basque_gdpcap"
)

echo "Downloading datasets into ${OUT_ROOT}"
PYTHONPATH=src python -m mcbench.cli fetch-dataset \
  --catalog-path "${CATALOG_PATH}" \
  --output-root "${OUT_ROOT}" \
  --skip-existing \
  --dataset-id "${DATASETS[@]}"

echo "Generating noisy benchmarks in ${NOISY_ROOT}"
for bench in gaussian_noise_ml100k sparse_corruption_ml100k one_bit_flip_ml100k; do
  PYTHONPATH=src python -m mcbench.cli make-noisy \
    --catalog-path "${CATALOG_PATH}" \
    --source-root "${OUT_ROOT}" \
    --output-root "${NOISY_ROOT}" \
    --benchmark-id "${bench}" \
    --seed 42
done

echo "Done."
