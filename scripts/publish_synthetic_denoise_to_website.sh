#!/usr/bin/env bash
set -euo pipefail

REPORT_DIR="${REPORT_DIR:-benchmarks/reports/synthetic_denoise}"
WEBSITE_DATA_DIR="${WEBSITE_DATA_DIR:-website/data}"

mkdir -p "${WEBSITE_DATA_DIR}"

# For website demo CSV loader compatibility.
cp "${REPORT_DIR}/synthetic_denoise_summary.csv" "${WEBSITE_DATA_DIR}/noise_sweep_results.csv"
cp "${REPORT_DIR}/synthetic_denoise_summary.csv" "${WEBSITE_DATA_DIR}/synthetic_denoise_results.csv"

if [[ -f "${REPORT_DIR}/nrmse_vs_noise.png" ]]; then
  cp "${REPORT_DIR}/nrmse_vs_noise.png" "${WEBSITE_DATA_DIR}/synthetic_denoise_nrmse_vs_noise.png"
fi
if [[ -f "${REPORT_DIR}/rmse_vs_noise.png" ]]; then
  cp "${REPORT_DIR}/rmse_vs_noise.png" "${WEBSITE_DATA_DIR}/synthetic_denoise_rmse_vs_noise.png"
fi

echo "Published synthetic denoise demo data to ${WEBSITE_DATA_DIR}"
echo "You can now click 'Load Demo CSV' on the website."

