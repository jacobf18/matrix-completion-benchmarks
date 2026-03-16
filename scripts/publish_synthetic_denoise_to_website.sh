#!/usr/bin/env bash
set -euo pipefail

REPORT_DIR="${REPORT_DIR:-benchmarks/reports/synthetic_denoise}"
WEBSITE_DATA_DIR="${WEBSITE_DATA_DIR:-website/data}"
TABIMPUTE_REPORT_DIR="${TABIMPUTE_REPORT_DIR:-}"

mkdir -p "${WEBSITE_DATA_DIR}"

TMP_CSV="$(mktemp)"
cp "${REPORT_DIR}/synthetic_denoise_summary.csv" "${TMP_CSV}"

if [[ -n "${TABIMPUTE_REPORT_DIR}" && -f "${TABIMPUTE_REPORT_DIR}/tabimpute_v2_summary.csv" ]]; then
  {
    cat "${TMP_CSV}"
    tail -n +2 "${TABIMPUTE_REPORT_DIR}/tabimpute_v2_summary.csv"
  } > "${TMP_CSV}.merged"
  mv "${TMP_CSV}.merged" "${TMP_CSV}"
fi

# For website demo CSV loader compatibility.
cp "${TMP_CSV}" "${WEBSITE_DATA_DIR}/noise_sweep_results.csv"
cp "${TMP_CSV}" "${WEBSITE_DATA_DIR}/synthetic_denoise_results.csv"
rm -f "${TMP_CSV}"

if [[ -f "${REPORT_DIR}/nrmse_vs_noise.png" ]]; then
  cp "${REPORT_DIR}/nrmse_vs_noise.png" "${WEBSITE_DATA_DIR}/synthetic_denoise_nrmse_vs_noise.png"
fi
if [[ -f "${REPORT_DIR}/rmse_vs_noise.png" ]]; then
  cp "${REPORT_DIR}/rmse_vs_noise.png" "${WEBSITE_DATA_DIR}/synthetic_denoise_rmse_vs_noise.png"
fi

echo "Published synthetic denoise demo data to ${WEBSITE_DATA_DIR}"
echo "You can now click 'Load Demo CSV' on the website."
