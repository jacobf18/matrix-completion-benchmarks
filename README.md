# Matrix Completion Benchmarks

Modular Python workflow for benchmarking matrix completion algorithms.

Website UI (lightweight static): `/Users/jfeit/matrix-completion-benchmarks/website/README.md`
Public deploy workflow: `/Users/jfeit/matrix-completion-benchmarks/.github/workflows/deploy-website.yml`

## Goals

- Accept any algorithm that maps `observed_matrix -> filled_matrix`.
- Keep algorithm execution separate from metric scoring.
- Make metrics easy to extend without touching algorithm code.
- Support reproducible dataset splits and run artifacts.

## Folder Structure

```text
matrix-completion-benchmarks/
├── benchmarks/
│   ├── datasets/
│   │   └── <dataset_name>/
│   │       ├── observed.npy
│   │       ├── ground_truth.npy
│   │       ├── eval_mask.npy
│   │       └── dataset_meta.json
│   ├── runs/
│   │   └── <dataset_name>/<algorithm_name>/
│   │       ├── prediction.npy
│   │       └── run_meta.json
│   └── reports/
│       └── <dataset_name>/<algorithm_name>_metrics.json
├── examples/
│   └── custom_plugin.py
└── src/mcbench/
    ├── algorithms/
    ├── metrics/
    ├── workflows/
    └── cli.py
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install imputation extras for `soft_impute`, `nuclear_norm_minimization`, `hyperimpute`, `missforest`, and `forest_diffusion`:

```bash
pip install -e '.[impute]'
```

Install tabular evaluation extras (downstream models + xgboost):

```bash
pip install -e '.[tabular]'
```

## Benchmark Catalog

The repository includes `/Users/jfeit/matrix-completion-benchmarks/benchmarks/catalog.yaml` with:

- Recommendation benchmarks:
  - `movielens_latest_small`
  - `movielens_100k`
  - `movielens_1m`
- Causal panel benchmarks:
  - `prop99_smoking` (California Proposition 99 smoking panel)
  - `basque_gdpcap`
- Noisy matrix completion presets:
  - `sim_lr_gaussian_low`
  - `sim_lr_gaussian_medium`
  - `sim_lr_gaussian_high`
  - `sim_orthogonal_student_t`
  - `sim_block_sparse_corrupt`

Nuclear-norm-focused benchmark index:

- `/Users/jfeit/matrix-completion-benchmarks/benchmarks/nnm_catalog.yaml`

List catalog entries:

```bash
mcbench list-datasets
```

Download datasets into `benchmarks/sources`:

```bash
mcbench fetch-dataset \
  --dataset-id movielens_latest_small movielens_100k movielens_1m prop99_smoking basque_gdpcap ckd_ehr_abu_dhabi \
  --output-root benchmarks/sources
```

Skip re-downloading datasets that are already prepared:

```bash
mcbench fetch-dataset \
  --dataset-id movielens_latest_small movielens_100k movielens_1m prop99_smoking basque_gdpcap \
  --output-root benchmarks/sources \
  --skip-existing
```

Force a refresh:

```bash
mcbench fetch-dataset \
  --dataset-id movielens_100k \
  --output-root benchmarks/sources \
  --force
```

Generate synthetic noisy benchmark matrices from catalog presets:

```bash
mcbench generate-simulated \
  --preset-id sim_lr_gaussian_medium \
  --output-root benchmarks/simulated \
  --seed 42
```

Override preset parameters to sweep noise levels or dimensions:

```bash
mcbench generate-simulated \
  --preset-id sim_lr_gaussian_medium \
  --params-json '{"sigma": 0.25, "rank": 15, "observed_fraction": 0.25}' \
  --output-root benchmarks/simulated \
  --seed 42
```

One-shot bash helper:

```bash
scripts/download_benchmarks.sh
```

## Synthetic Noise Sweep (Global Mean vs SoftImpute)

Run a Gaussian noise-level sweep on simulated low-rank data:

```bash
PYTHONPATH=src python scripts/run_synthetic_noise_sweep.py \
  --noise-levels 0.05,0.1,0.2,0.35,0.5 \
  --algorithms global_mean soft_impute
```

This writes:
- `benchmarks/reports/noise_sweep/noise_sweep_results.csv`

Plot normalized RMSE against noise level:

```bash
PYTHONPATH=src python scripts/plot_noise_sweep.py \
  --results-csv benchmarks/reports/noise_sweep/noise_sweep_results.csv \
  --metric nrmse \
  --output-path benchmarks/reports/noise_sweep/nrmse_vs_noise.png
```

For RMSE instead:

```bash
PYTHONPATH=src python scripts/plot_noise_sweep.py \
  --results-csv benchmarks/reports/noise_sweep/noise_sweep_results.csv \
  --metric rmse \
  --output-path benchmarks/reports/noise_sweep/rmse_vs_noise.png
```

## End-to-End Synthetic Denoising Benchmark

Run implemented algorithms on the synthetic noisy matrix-completion presets (denoising target = clean ground truth):

```bash
scripts/run_synthetic_denoise_e2e.sh
```

Choose a custom algorithm subset:

```bash
scripts/run_synthetic_denoise_e2e.sh \
  --algorithms "global_mean soft_impute missforest"
```

You can also set seeds/presets:

```bash
scripts/run_synthetic_denoise_e2e.sh \
  --seeds "0,1,2,3" \
  --preset-ids "sim_lr_gaussian_low sim_lr_gaussian_medium" \
  --algorithms "global_mean row_mean soft_impute"
```

Outputs:

- `benchmarks/reports/synthetic_denoise/synthetic_denoise_detailed.csv`
- `benchmarks/reports/synthetic_denoise/synthetic_denoise_summary.csv`
- `benchmarks/reports/synthetic_denoise/synthetic_denoise_summary.json`
- `benchmarks/reports/synthetic_denoise/nrmse_vs_noise.png`
- `benchmarks/reports/synthetic_denoise/rmse_vs_noise.png`

Publish summary for website demo loading:

```bash
scripts/publish_synthetic_denoise_to_website.sh
```

## Hankel Matrix Completion Benchmarks

Hankel benchmarks target time-series recovery via low-rank structure in the trajectory (Hankel) matrix.

List Hankel presets:

```bash
PYTHONPATH=src python -m mcbench.cli list-datasets
```

Run Hankel benchmark presets with multiple methods:

```bash
PYTHONPATH=src python scripts/run_hankel_benchmarks.py \
  --preset-ids hankel_gaussian_sigma_0p01 hankel_gaussian_sigma_0p05 hankel_gaussian_sigma_0p10 \
  --algorithms global_mean soft_impute cadzow \
  --output-dir benchmarks/reports/hankel
```

This writes:

- `benchmarks/reports/hankel/hankel_results.csv`

Plot quality vs noise level:

```bash
PYTHONPATH=src python scripts/plot_hankel_results.py \
  --results-csv benchmarks/reports/hankel/hankel_results.csv \
  --metric nrmse_missing \
  --x-axis noise_sigma \
  --output-path benchmarks/reports/hankel/nrmse_missing_vs_noise.png
```

## Workflow

### 1) Prepare a benchmark dataset

Input can be `.npy`, `.csv`, or `.tsv`.  
`prepare-split` hides a random fraction of known cells to create an evaluation mask.

```bash
mcbench prepare-split \
  --input-matrix benchmarks/sources/movielens_100k/matrix.npy \
  --output-dataset-dir benchmarks/datasets/movie_lens_small \
  --holdout-fraction 0.2 \
  --seed 42
```

Output files:
- `observed.npy`: matrix with held-out cells set to `NaN`.
- `ground_truth.npy`: original matrix values.
- `eval_mask.npy`: boolean mask of held-out cells to score.

### 2) Run an algorithm (independent step)

```bash
mcbench run-algorithm \
  --dataset-dir benchmarks/datasets/movie_lens_small \
  --algorithm row_mean \
  --output-dir benchmarks/runs/movie_lens_small/row_mean
```

This creates:
- `prediction.npy`: completed matrix from that algorithm.
- `run_meta.json`: runtime and parameter metadata.

### 3) Evaluate metrics later (separate step)

```bash
mcbench evaluate \
  --dataset-dir benchmarks/datasets/movie_lens_small \
  --prediction-path benchmarks/runs/movie_lens_small/row_mean/prediction.npy \
  --metrics rmse mae nmae \
  --output-path benchmarks/reports/movie_lens_small/row_mean_metrics.json
```

This separation lets you run slow algorithms once, then evaluate many metric sets afterward.

## Built-in Algorithms

- `global_mean`
- `row_mean`
- `soft_impute`
- `nuclear_norm_minimization`
- `hyperimpute`
- `missforest`
- `forest_diffusion`

List all available algorithms:

```bash
mcbench list-algorithms
```

`soft_impute` parameters (backed by `fancyimpute.SoftImpute`) can be passed with `--params-json`, for example:

```bash
mcbench run-algorithm \
  --dataset-dir benchmarks/datasets/movie_lens_small \
  --algorithm soft_impute \
  --params-json '{"shrinkage": 0.8, "max_iters": 200, "tol": 1e-6, "rank": 20, "init_fill": "global_mean"}' \
  --output-dir benchmarks/runs/movie_lens_small/soft_impute
```

`nuclear_norm_minimization` parameters (backed by `fancyimpute.NuclearNormMinimization`) example:

```bash
mcbench run-algorithm \
  --dataset-dir benchmarks/datasets/movie_lens_small \
  --algorithm nuclear_norm_minimization \
  --params-json '{"max_iters": 5000, "error_tolerance": 1e-7, "min_value": 0.0, "max_value": 5.0}' \
  --output-dir benchmarks/runs/movie_lens_small/nuclear_norm_minimization
```

`hyperimpute` and `missforest` (both via `hyperimpute.plugins.imputers`) examples:

```bash
mcbench run-algorithm \
  --dataset-dir benchmarks/datasets/movie_lens_small \
  --algorithm hyperimpute \
  --params-json '{}' \
  --output-dir benchmarks/runs/movie_lens_small/hyperimpute

mcbench run-algorithm \
  --dataset-dir benchmarks/datasets/movie_lens_small \
  --algorithm missforest \
  --params-json '{}' \
  --output-dir benchmarks/runs/movie_lens_small/missforest
```

`forest_diffusion` (via `ForestDiffusion.ForestDiffusionModel`) example:

```bash
mcbench run-algorithm \
  --dataset-dir benchmarks/datasets/movie_lens_small \
  --algorithm forest_diffusion \
  --params-json '{"n_t": 25, "model": "xgboost", "diffusion_type": "vp", "n_estimators": 100, "max_depth": 7, "k": 1}' \
  --output-dir benchmarks/runs/movie_lens_small/forest_diffusion
```

## Built-in Metrics

- `rmse`
- `mae`
- `nmae`

List all available metrics:

```bash
mcbench list-metrics
```

Tabular downstream metrics in registry:

- `downstream_accuracy_linear`
- `downstream_accuracy_random_forest`
- `downstream_accuracy_xgboost`
- `downstream_balanced_accuracy_*` (linear/random_forest/xgboost)
- `downstream_f1_*` (linear/random_forest/xgboost)
- `downstream_roc_auc_*` (linear/random_forest/xgboost, binary classification)
- `downstream_average_precision_*` (linear/random_forest/xgboost, binary classification)

## Add Your Own Algorithm and Metric

Create a plugin module that registers implementations (see `examples/custom_plugin.py`), then load it with `--plugin`.

```bash
mcbench --plugin examples.custom_plugin list-algorithms
mcbench --plugin examples.custom_plugin list-metrics
```

Run with a plugin algorithm:

```bash
mcbench --plugin examples.custom_plugin run-algorithm \
  --dataset-dir benchmarks/datasets/movie_lens_small \
  --algorithm column_mean \
  --output-dir benchmarks/runs/movie_lens_small/column_mean
```

Evaluate with a plugin metric:

```bash
mcbench --plugin examples.custom_plugin evaluate \
  --dataset-dir benchmarks/datasets/movie_lens_small \
  --prediction-path benchmarks/runs/movie_lens_small/column_mean/prediction.npy \
  --metrics rmse max_abs_error \
  --output-path benchmarks/reports/movie_lens_small/column_mean_metrics.json
```

## API Contracts

Algorithm contract (`MatrixCompletionAlgorithm`):
- Input: 2D `numpy` matrix with `NaN` at missing entries.
- Output: 2D `numpy` matrix of identical shape, with filled values.

Metric contract (`MatrixMetric`):
- Input: `y_true`, `y_pred`, and `eval_mask`.
- Output: one scalar float.

## Notes

- The framework is domain-agnostic and can be used for recommendation matrices, causal panel matrices, tabular missingness, and noisy matrix completion.
- For real data where full ground truth is not available, create dataset bundles manually with your own `eval_mask` policy.

## Tabular Imputation Benchmarks

### CKD EHR (Kaggle) End-to-End Regression Benchmark

Requirements:

- Install extras: `pip install -e '.[tabular,impute]'`
- Install and authenticate Kaggle CLI (`kaggle.json`) to access:
  `https://www.kaggle.com/datasets/davidechicco/chronic-kidney-disease-ehrs-abu-dhabi`

Fetch the source dataset:

```bash
PYTHONPATH=src python -m mcbench.cli fetch-dataset \
  --dataset-id ckd_ehr_abu_dhabi \
  --output-root benchmarks/sources \
  --skip-existing
```

Prepare tabular bundles with multiple missingness patterns/fractions:

```bash
PYTHONPATH=src python scripts/prepare_ckd_ehr_tabular_benchmark.py \
  --source-dir benchmarks/sources/ckd_ehr_abu_dhabi/raw \
  --output-root benchmarks/datasets/ckd_ehr_regression \
  --target-column age \
  --patterns mcar,mar_logistic,mnar_self_logistic,block,bursty \
  --missing-fractions 0.1,0.2,0.4 \
  --seeds 0,1,2 \
  --test-fraction 0.2
```

Run imputation + multiple-imputation evaluation + downstream regression:

```bash
PYTHONPATH=src python scripts/run_ckd_ehr_regression_benchmark.py \
  --dataset-root benchmarks/datasets/ckd_ehr_regression \
  --output-root benchmarks/reports/ckd_ehr_regression \
  --algorithms global_mean,row_mean,soft_impute \
  --num-imputations 5
```

To include the `mi_gaussian` multiple-imputation baseline (opt-in):

```bash
PYTHONPATH=src python scripts/run_ckd_ehr_regression_benchmark.py \
  --dataset-root benchmarks/datasets/ckd_ehr_regression \
  --output-root benchmarks/reports/ckd_ehr_regression \
  --algorithms global_mean,row_mean,soft_impute \
  --num-imputations 5 \
  --include-mi-gaussian
```

The script writes per-bundle JSON outputs and a summary CSV:

- `benchmarks/reports/ckd_ehr_regression/ckd_ehr_regression_summary.csv`

Prepare a tabular benchmark bundle with downstream train/test row masks:

```bash
PYTHONPATH=src python scripts/prepare_tabular_benchmark.py \
  --input-matrix /path/to/tabular_full.npy \
  --output-dataset-dir benchmarks/datasets/tabular_demo \
  --target-col 0 \
  --missing-fraction 0.2 \
  --test-fraction 0.2 \
  --seed 42
```

Run a single imputation method and evaluate cellwise + downstream metrics:

```bash
PYTHONPATH=src python -m mcbench.cli run-algorithm \
  --dataset-dir benchmarks/datasets/tabular_demo \
  --algorithm soft_impute \
  --output-dir benchmarks/runs/tabular_demo/soft_impute

PYTHONPATH=src python scripts/evaluate_tabular_imputation.py \
  --dataset-dir benchmarks/datasets/tabular_demo \
  --prediction-path benchmarks/runs/tabular_demo/soft_impute/prediction.npy \
  --task classification \
  --output-path benchmarks/reports/tabular_demo/soft_impute_eval.json
```

Multiple-imputation benchmark (Gaussian posterior-style baseline):

```bash
PYTHONPATH=src python scripts/generate_multiple_imputations.py \
  --dataset-dir benchmarks/datasets/tabular_demo \
  --num-imputations 5 \
  --seed 42 \
  --output-dir benchmarks/runs/tabular_demo/mi_gaussian
```

Evaluate multiple imputations:

```bash
PYTHONPATH=src python scripts/evaluate_tabular_imputation.py \
  --dataset-dir benchmarks/datasets/tabular_demo \
  --prediction-path benchmarks/runs/tabular_demo/mi_gaussian/prediction_000.npy \
  --prediction-path benchmarks/runs/tabular_demo/mi_gaussian/prediction_001.npy \
  --prediction-path benchmarks/runs/tabular_demo/mi_gaussian/prediction_002.npy \
  --prediction-path benchmarks/runs/tabular_demo/mi_gaussian/prediction_003.npy \
  --prediction-path benchmarks/runs/tabular_demo/mi_gaussian/prediction_004.npy \
  --task classification \
  --output-path benchmarks/reports/tabular_demo/mi_gaussian_eval.json
```

## Missingness Pattern Generation

Generate reusable masks/observed matrices independently of algorithms:

```bash
PYTHONPATH=src python scripts/generate_missingness_pattern.py \
  --input-matrix /path/to/full_matrix.npy \
  --output-dir benchmarks/datasets/missingness_demo \
  --pattern mar_logistic \
  --missing-fraction 0.3 \
  --feature-col 0 \
  --seed 42
```

Supported patterns:

- `mcar`
- `mar_logistic`
- `mnar_self_logistic`
- `block`
- `bursty`
