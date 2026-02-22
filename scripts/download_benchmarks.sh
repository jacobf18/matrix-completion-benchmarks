#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/download_benchmarks.sh

CATALOG_PATH="${CATALOG_PATH:-benchmarks/catalog.yaml}"
OUT_ROOT="${OUT_ROOT:-benchmarks/sources}"
SIM_ROOT="${SIM_ROOT:-benchmarks/simulated}"

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

echo "Generating synthetic noisy benchmarks in ${SIM_ROOT}"
for preset in sim_lr_gaussian_low sim_lr_gaussian_medium sim_lr_gaussian_high sim_orthogonal_student_t sim_block_sparse_corrupt; do
  if ! PYTHONPATH=src python -m mcbench.cli generate-simulated \
    --catalog-path "${CATALOG_PATH}" \
    --output-root "${SIM_ROOT}" \
    --preset-id "${preset}" \
    --seed 42; then
    echo "warning: failed to generate synthetic benchmark ${preset}; continuing"
  fi
done

echo "Done."
