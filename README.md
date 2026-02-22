# Matrix Completion Benchmarks

Modular Python workflow for benchmarking matrix completion algorithms.

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

List catalog entries:

```bash
mcbench list-datasets
```

Download datasets into `benchmarks/sources`:

```bash
mcbench fetch-dataset \
  --dataset-id movielens_latest_small movielens_100k movielens_1m prop99_smoking basque_gdpcap \
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
