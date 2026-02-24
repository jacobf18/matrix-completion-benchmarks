#!/usr/bin/env bash
set -euo pipefail

CATALOG_PATH="${CATALOG_PATH:-benchmarks/catalog.yaml}"
DATASET_ROOT="${DATASET_ROOT:-benchmarks/simulated/denoise_runs}"
REPORT_DIR="${REPORT_DIR:-benchmarks/reports/synthetic_denoise}"
SEEDS="${SEEDS:-0,1,2}"
ALGORITHMS="${ALGORITHMS:-global_mean row_mean soft_impute nuclear_norm_minimization hyperimpute missforest forest_diffusion}"

echo "Running synthetic denoising benchmark..."
PYTHONPATH=src python scripts/run_synthetic_denoise_benchmark.py \
  --catalog-path "${CATALOG_PATH}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${REPORT_DIR}" \
  --seeds "${SEEDS}" \
  --algorithms ${ALGORITHMS}

echo "Plotting NRMSE vs noise_sigma..."
PYTHONPATH=src python scripts/plot_noise_sweep.py \
  --results-csv "${REPORT_DIR}/synthetic_denoise_summary.csv" \
  --metric nrmse \
  --output-path "${REPORT_DIR}/nrmse_vs_noise.png"

echo "Plotting RMSE vs noise_sigma..."
PYTHONPATH=src python scripts/plot_noise_sweep.py \
  --results-csv "${REPORT_DIR}/synthetic_denoise_summary.csv" \
  --metric rmse \
  --output-path "${REPORT_DIR}/rmse_vs_noise.png"

echo "Done. Results in ${REPORT_DIR}"

