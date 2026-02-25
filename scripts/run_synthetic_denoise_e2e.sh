#!/usr/bin/env bash
set -euo pipefail

CATALOG_PATH="${CATALOG_PATH:-benchmarks/catalog.yaml}"
DATASET_ROOT="${DATASET_ROOT:-benchmarks/simulated/denoise_runs}"
REPORT_DIR="${REPORT_DIR:-benchmarks/reports/synthetic_denoise}"
SEEDS="${SEEDS:-0,1,2}"
ALGORITHMS="${ALGORITHMS:-global_mean row_mean soft_impute nuclear_norm_minimization hyperimpute missforest forest_diffusion}"
PRESET_IDS="${PRESET_IDS:-sim_lr_gaussian_sigma_0p01 sim_lr_gaussian_sigma_0p03 sim_lr_gaussian_sigma_0p1 sim_lr_gaussian_sigma_0p3 sim_lr_gaussian_sigma_1p0 sim_lr_gaussian_sigma_3p0}"
ALGO_PARAMS_JSON="${ALGO_PARAMS_JSON:-{}}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-false}"
SYNC_WEBSITE_DATA="${SYNC_WEBSITE_DATA:-true}"

usage() {
  cat <<'EOF'
Usage: scripts/run_synthetic_denoise_e2e.sh [options]

Options:
  --algorithms "a b c"       Space-separated algorithm list.
  --seeds "0,1,2"            Comma-separated random seeds.
  --preset-ids "p1 p2"       Optional space-separated synthetic preset IDs.
  --catalog-path PATH        Catalog path (default: benchmarks/catalog.yaml).
  --dataset-root PATH        Dataset output root.
  --report-dir PATH          Report output root.
  --algo-params-json JSON    Per-algorithm params JSON object.
  --save-predictions         Save prediction matrices for each run.
  --no-sync-website-data     Skip syncing website/data/synthetic_denoise_results.csv.
  -h, --help                 Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --algorithms)
      ALGORITHMS="$2"
      shift 2
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --preset-ids)
      PRESET_IDS="$2"
      shift 2
      ;;
    --catalog-path)
      CATALOG_PATH="$2"
      shift 2
      ;;
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --report-dir)
      REPORT_DIR="$2"
      shift 2
      ;;
    --algo-params-json)
      ALGO_PARAMS_JSON="$2"
      shift 2
      ;;
    --save-predictions)
      SAVE_PREDICTIONS="true"
      shift
      ;;
    --no-sync-website-data)
      SYNC_WEBSITE_DATA="false"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

echo "Running synthetic denoising benchmark..."
read -r -a ALGO_ARR <<< "${ALGORITHMS}"
CMD=(
  python scripts/run_synthetic_denoise_benchmark.py
  --catalog-path "${CATALOG_PATH}"
  --dataset-root "${DATASET_ROOT}"
  --output-dir "${REPORT_DIR}"
  --seeds "${SEEDS}"
  --algorithm-params-json "${ALGO_PARAMS_JSON}"
  --algorithms "${ALGO_ARR[@]}"
)

if [[ -n "${PRESET_IDS}" ]]; then
  read -r -a PRESET_ARR <<< "${PRESET_IDS}"
  CMD+=(--preset-ids "${PRESET_ARR[@]}")
fi
if [[ "${SAVE_PREDICTIONS}" == "true" ]]; then
  CMD+=(--save-predictions)
fi

PYTHONPATH=src "${CMD[@]}"

echo "Plotting NRMSE vs noise_sigma..."
PYTHONPATH=src python scripts/plot_noise_sweep.py \
  --results-csv "${REPORT_DIR}/synthetic_denoise_summary.csv" \
  --metric nrmse \
  --x-log \
  --output-path "${REPORT_DIR}/nrmse_vs_noise.png"

echo "Plotting RMSE vs noise_sigma..."
PYTHONPATH=src python scripts/plot_noise_sweep.py \
  --results-csv "${REPORT_DIR}/synthetic_denoise_summary.csv" \
  --metric rmse \
  --x-log \
  --output-path "${REPORT_DIR}/rmse_vs_noise.png"

if [[ "${SYNC_WEBSITE_DATA}" == "true" && -d "website/data" ]]; then
  cp "${REPORT_DIR}/synthetic_denoise_summary.csv" "website/data/synthetic_denoise_results.csv"
  echo "Synced website data: website/data/synthetic_denoise_results.csv"
fi

echo "Done. Results in ${REPORT_DIR}"
