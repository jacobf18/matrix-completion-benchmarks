#!/usr/bin/env bash
set -euo pipefail

# End-to-end CKD EHR benchmark runner across all implemented algorithms.
#
# Usage:
#   scripts/run_ckd_all_algorithms.sh [options]
#
# Example:
#   scripts/run_ckd_all_algorithms.sh \
#     --target-column EventCKD35 \
#     --task classification \
#     --patterns mcar,mar_logistic \
#     --missing-fractions 0.1,0.2,0.3 \
#     --seeds 0,1,2

SOURCE_DIR="${SOURCE_DIR:-benchmarks/sources/ckd_ehr_abu_dhabi/raw}"
DATASET_ROOT="${DATASET_ROOT:-benchmarks/datasets/ckd_ehr_all_algorithms}"
REPORT_ROOT="${REPORT_ROOT:-benchmarks/reports/ckd_ehr_all_algorithms}"
TARGET_COLUMN="${TARGET_COLUMN:-EventCKD35}"
TASK="${TASK:-classification}"
PATTERNS="${PATTERNS:-mcar,mar_logistic}"
MISSING_FRACTIONS="${MISSING_FRACTIONS:-0.1,0.2,0.3,0.4,0.5,0.6,0.7}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
TEST_FRACTION="${TEST_FRACTION:-0.2}"
INCLUDE_MI_GAUSSIAN="${INCLUDE_MI_GAUSSIAN:-false}"

ALGORITHMS="${ALGORITHMS:-global_mean,row_mean,soft_impute,nuclear_norm_minimization,hyperimpute,missforest,forest_diffusion,tab_impute}"
ALGORITHM_PARAMS_JSON="${ALGORITHM_PARAMS_JSON:-{\"tab_impute\":{\"device\":\"cpu\",\"max_num_rows\":256,\"max_num_chunks\":4,\"num_repeats\":1}}}"

usage() {
  cat <<'EOF'
Usage: scripts/run_ckd_all_algorithms.sh [options]

Options:
  --source-dir PATH              CKD raw source directory.
  --dataset-root PATH            Output root for generated benchmark bundles.
  --report-root PATH             Output root for benchmark reports.
  --target-column NAME           Target column (EventCKD35 or TimeToEventMonths, etc.).
  --task NAME                    classification or regression.
  --patterns CSV                 Missingness patterns CSV (e.g. mcar,mar_logistic).
  --missing-fractions CSV        Missingness fractions CSV.
  --seeds CSV                    Seed list CSV.
  --test-fraction FLOAT          Downstream test split fraction.
  --algorithms CSV               Algorithms CSV list.
  --algorithm-params-json JSON   JSON object keyed by algorithm name.
  --include-mi-gaussian          Include mi_gaussian baseline rows.
  -h, --help                     Show help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-dir) SOURCE_DIR="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --report-root) REPORT_ROOT="$2"; shift 2 ;;
    --target-column) TARGET_COLUMN="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --patterns) PATTERNS="$2"; shift 2 ;;
    --missing-fractions) MISSING_FRACTIONS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --test-fraction) TEST_FRACTION="$2"; shift 2 ;;
    --algorithms) ALGORITHMS="$2"; shift 2 ;;
    --algorithm-params-json) ALGORITHM_PARAMS_JSON="$2"; shift 2 ;;
    --include-mi-gaussian) INCLUDE_MI_GAUSSIAN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

echo "Preparing CKD benchmark bundles..."
PYTHONPATH=src python scripts/prepare_ckd_ehr_tabular_benchmark.py \
  --source-dir "${SOURCE_DIR}" \
  --output-root "${DATASET_ROOT}" \
  --target-column "${TARGET_COLUMN}" \
  --patterns "${PATTERNS}" \
  --missing-fractions "${MISSING_FRACTIONS}" \
  --seeds "${SEEDS}" \
  --test-fraction "${TEST_FRACTION}"

echo "Running algorithms: ${ALGORITHMS}"
RUN_CMD=(
  python scripts/run_ckd_ehr_regression_benchmark.py
  --dataset-root "${DATASET_ROOT}"
  --output-root "${REPORT_ROOT}"
  --algorithms "${ALGORITHMS}"
  --task "${TASK}"
  --algorithm-params-json "${ALGORITHM_PARAMS_JSON}"
)

if [[ "${INCLUDE_MI_GAUSSIAN}" == "true" ]]; then
  RUN_CMD+=(--include-mi-gaussian)
fi

PYTHONPATH=src "${RUN_CMD[@]}"

echo "Done."
echo "Summary CSV: ${REPORT_ROOT}/ckd_ehr_regression_summary.csv"
