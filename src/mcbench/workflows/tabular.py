from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..io import load_matrix, save_json, save_mask, save_matrix


def prepare_tabular_benchmark(
    input_matrix_path: Path,
    output_dataset_dir: Path,
    target_col: int,
    missing_fraction: float,
    test_fraction: float,
    seed: int,
) -> None:
    matrix = load_matrix(input_matrix_path)
    if matrix.ndim != 2:
        raise ValueError("Input matrix must be 2D.")
    n_rows, n_cols = matrix.shape
    if not (0 <= target_col < n_cols):
        raise ValueError("target_col out of bounds.")
    if not (0 < missing_fraction < 1):
        raise ValueError("missing_fraction must be in (0, 1).")
    if not (0 < test_fraction < 1):
        raise ValueError("test_fraction must be in (0, 1).")

    rng = np.random.default_rng(seed)
    observed = matrix.astype(np.float64, copy=True)
    finite = np.isfinite(observed)

    row_perm = rng.permutation(n_rows)
    n_test = max(1, int(round(n_rows * test_fraction)))
    test_rows = row_perm[:n_test]
    train_rows = row_perm[n_test:]

    train_mask = np.zeros(n_rows, dtype=bool)
    test_mask = np.zeros(n_rows, dtype=bool)
    train_mask[train_rows] = True
    test_mask[test_rows] = True

    # Evaluate imputation only on feature columns.
    feature_mask = np.ones_like(observed, dtype=bool)
    feature_mask[:, target_col] = False
    candidate_mask = finite & feature_mask
    candidate_idx = np.flatnonzero(candidate_mask)
    n_eval = max(1, int(round(candidate_idx.size * missing_fraction)))
    eval_idx = rng.choice(candidate_idx, size=n_eval, replace=False)
    eval_mask = np.zeros_like(observed, dtype=bool)
    eval_mask.flat[eval_idx] = True
    observed.flat[eval_idx] = np.nan

    output_dataset_dir.mkdir(parents=True, exist_ok=True)
    save_matrix(output_dataset_dir / "ground_truth.npy", matrix)
    save_matrix(output_dataset_dir / "observed.npy", observed)
    save_mask(output_dataset_dir / "eval_mask.npy", eval_mask)
    save_mask(output_dataset_dir / "downstream_train_mask.npy", train_mask)
    save_mask(output_dataset_dir / "downstream_test_mask.npy", test_mask)
    save_json(
        output_dataset_dir / "dataset_meta.json",
        {
            "input_matrix_path": str(input_matrix_path),
            "target_col": target_col,
            "missing_fraction": missing_fraction,
            "test_fraction": test_fraction,
            "seed": seed,
            "shape": [n_rows, n_cols],
            "eval_count": int(np.sum(eval_mask)),
            "train_rows": int(np.sum(train_mask)),
            "test_rows": int(np.sum(test_mask)),
        },
    )


def generate_multiple_imputations_gaussian(
    observed: np.ndarray,
    num_imputations: int,
    seed: int,
    min_std: float = 1e-6,
) -> list[np.ndarray]:
    if num_imputations < 2:
        raise ValueError("num_imputations must be >= 2 for multiple imputation.")
    out: list[np.ndarray] = []
    finite = np.isfinite(observed)
    col_means = np.nanmean(observed, axis=0)
    col_stds = np.nanstd(observed, axis=0)
    col_stds = np.where(np.isfinite(col_stds), np.maximum(col_stds, min_std), 1.0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)

    # Infer simple per-column sampling families from observed values.
    col_kind: list[str] = []
    col_min = np.nanmin(observed, axis=0)
    col_max = np.nanmax(observed, axis=0)
    col_min = np.where(np.isfinite(col_min), col_min, 0.0)
    col_max = np.where(np.isfinite(col_max), col_max, 1.0)
    col_bernoulli_p = np.clip(col_means, 1e-4, 1 - 1e-4)
    col_empirical_vals: list[np.ndarray] = []
    col_empirical_probs: list[np.ndarray] = []

    for j in range(observed.shape[1]):
        vals = observed[:, j]
        finite_vals = vals[np.isfinite(vals)]
        if finite_vals.size == 0:
            col_kind.append("normal")
            col_empirical_vals.append(np.array([], dtype=np.float64))
            col_empirical_probs.append(np.array([], dtype=np.float64))
            continue
        uniq = np.unique(finite_vals)
        if uniq.size <= 2 and np.all(np.isin(uniq, [0.0, 1.0])):
            col_kind.append("bernoulli")
            col_empirical_vals.append(np.array([], dtype=np.float64))
            col_empirical_probs.append(np.array([], dtype=np.float64))
            continue
        # Treat very low-cardinality integer-like columns as categorical.
        if uniq.size <= 8 and np.allclose(finite_vals, np.round(finite_vals), atol=1e-8):
            counts = np.array([(finite_vals == u).sum() for u in uniq], dtype=np.float64)
            probs = counts / np.sum(counts)
            col_kind.append("categorical")
            col_empirical_vals.append(uniq.astype(np.float64))
            col_empirical_probs.append(probs)
            continue
        col_kind.append("normal")
        col_empirical_vals.append(np.array([], dtype=np.float64))
        col_empirical_probs.append(np.array([], dtype=np.float64))

    for idx in range(num_imputations):
        rng = np.random.default_rng(seed + idx)
        pred = observed.astype(np.float64, copy=True)
        miss_r, miss_c = np.where(~finite)
        if miss_r.size > 0:
            samples = np.empty(miss_r.size, dtype=np.float64)
            for k, col in enumerate(miss_c):
                kind = col_kind[col]
                if kind == "bernoulli":
                    samples[k] = float(rng.binomial(1, col_bernoulli_p[col]))
                elif kind == "categorical":
                    vals = col_empirical_vals[col]
                    probs = col_empirical_probs[col]
                    if vals.size == 0:
                        samples[k] = col_means[col]
                    else:
                        samples[k] = float(rng.choice(vals, p=probs))
                else:
                    draw = float(rng.normal(loc=col_means[col], scale=col_stds[col]))
                    # Keep draws in a plausible support range for the column.
                    samples[k] = float(np.clip(draw, col_min[col], col_max[col]))
            pred[miss_r, miss_c] = samples
        out.append(pred)
    return out


def evaluate_single_imputation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eval_mask: np.ndarray,
) -> dict[str, float]:
    valid = eval_mask & np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        raise ValueError("No valid eval cells.")
    t = y_true[valid]
    p = y_pred[valid]
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    mae = float(np.mean(np.abs(p - t)))
    denom = float(np.max(t) - np.min(t))
    nrmse = 0.0 if denom == 0 else rmse / denom
    nmae = 0.0 if denom == 0 else mae / denom
    return {"rmse": rmse, "mae": mae, "nrmse": nrmse, "nmae": nmae}


def evaluate_downstream_models(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_col: int,
    train_row_mask: np.ndarray,
    test_row_mask: np.ndarray,
    task: str = "classification",
) -> dict[str, float]:
    x_train, y_train, x_test, y_test = _build_supervised_splits(
        y_true=y_true,
        y_pred=y_pred,
        target_col=target_col,
        train_row_mask=train_row_mask,
        test_row_mask=test_row_mask,
    )
    return _fit_and_score_models(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, task=task)


def evaluate_multiple_imputation_metrics(
    y_true: np.ndarray,
    y_preds: list[np.ndarray],
    eval_mask: np.ndarray,
    target_col: int,
    train_row_mask: np.ndarray,
    test_row_mask: np.ndarray,
    task: str = "classification",
) -> dict[str, float]:
    if len(y_preds) < 2:
        raise ValueError("Need at least two imputations for multiple-imputation metrics.")

    base_scores = [evaluate_single_imputation_metrics(y_true=y_true, y_pred=pred, eval_mask=eval_mask) for pred in y_preds]
    out: dict[str, float] = {}
    for key in ("rmse", "mae", "nrmse", "nmae"):
        vals = np.array([scores[key] for scores in base_scores], dtype=np.float64)
        out[f"{key}_mi_mean"] = float(np.mean(vals))
        out[f"{key}_mi_std"] = float(np.std(vals))
        out[f"{key}_mi_between_var"] = float(np.var(vals, ddof=1)) if vals.size > 1 else 0.0

    downstream_scores = []
    model_preds = {"linear": [], "random_forest": [], "xgboost": []}
    model_score_preds = {"linear": [], "random_forest": [], "xgboost": []}
    for pred in y_preds:
        x_train, y_train, x_test, y_test = _build_supervised_splits(
            y_true=y_true,
            y_pred=pred,
            target_col=target_col,
            train_row_mask=train_row_mask,
            test_row_mask=test_row_mask,
        )
        score_map, pred_map, score_pred_map = _fit_and_score_models_with_predictions(
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, task=task
        )
        downstream_scores.append(score_map)
        for model_name, model_pred in pred_map.items():
            if model_pred is not None:
                model_preds[model_name].append(model_pred)
        for model_name, score_pred in score_pred_map.items():
            if score_pred is not None:
                model_score_preds[model_name].append(score_pred)

    if task == "classification":
        metric_keys = ("accuracy", "balanced_accuracy", "f1", "roc_auc", "average_precision")
    else:
        metric_keys = ("r2",)

    y_test_ref = _build_supervised_splits(
        y_true=y_true,
        y_pred=y_preds[0],
        target_col=target_col,
        train_row_mask=train_row_mask,
        test_row_mask=test_row_mask,
    )[3]

    for model_name in ("linear", "random_forest", "xgboost"):
        for key in metric_keys:
            vals = np.array(
                [
                    scores[f"downstream_{key}_{model_name}"]
                    for scores in downstream_scores
                    if f"downstream_{key}_{model_name}" in scores and np.isfinite(scores[f"downstream_{key}_{model_name}"])
                ],
                dtype=np.float64,
            )
            if vals.size == 0:
                continue
            out[f"downstream_{key}_{model_name}_mi_mean"] = float(np.mean(vals))
            out[f"downstream_{key}_{model_name}_mi_std"] = float(np.std(vals))

        if task == "classification":
            pooled_labels = _pool_predictions(model_preds[model_name], task=task)
            if pooled_labels is not None:
                out[f"downstream_accuracy_{model_name}_mi_pooled"] = _score_from_predictions(
                    y_true=y_test_ref,
                    y_pred=pooled_labels,
                    task=task,
                    metric_key="accuracy",
                )
                out[f"downstream_balanced_accuracy_{model_name}_mi_pooled"] = _score_from_predictions(
                    y_true=y_test_ref,
                    y_pred=pooled_labels,
                    task=task,
                    metric_key="balanced_accuracy",
                )
                out[f"downstream_f1_{model_name}_mi_pooled"] = _score_from_predictions(
                    y_true=y_test_ref,
                    y_pred=pooled_labels,
                    task=task,
                    metric_key="f1",
                )
            pooled_scores = _pool_score_predictions(model_score_preds[model_name])
            if pooled_scores is not None:
                out[f"downstream_roc_auc_{model_name}_mi_pooled"] = _score_from_predictions(
                    y_true=y_test_ref,
                    y_pred=pooled_scores,
                    task=task,
                    metric_key="roc_auc",
                )
                out[f"downstream_average_precision_{model_name}_mi_pooled"] = _score_from_predictions(
                    y_true=y_test_ref,
                    y_pred=pooled_scores,
                    task=task,
                    metric_key="average_precision",
                )
        else:
            pooled = _pool_predictions(model_preds[model_name], task=task)
            if pooled is not None:
                out[f"downstream_r2_{model_name}_mi_pooled"] = _score_from_predictions(
                    y_true=y_test_ref,
                    y_pred=pooled,
                    task=task,
                    metric_key="r2",
                )
    return out


def _build_supervised_splits(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_col: int,
    train_row_mask: np.ndarray,
    test_row_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred shape mismatch.")
    n_rows, n_cols = y_true.shape
    if not (0 <= target_col < n_cols):
        raise ValueError("target_col out of bounds.")
    if train_row_mask.shape[0] != n_rows or test_row_mask.shape[0] != n_rows:
        raise ValueError("train/test row mask size mismatch.")

    feature_cols = [c for c in range(n_cols) if c != target_col]
    x = y_pred[:, feature_cols]
    y = y_true[:, target_col]
    train_valid = train_row_mask & np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    test_valid = test_row_mask & np.isfinite(y) & np.all(np.isfinite(x), axis=1)

    x_train = x[train_valid]
    y_train = y[train_valid]
    x_test = x[test_valid]
    y_test = y[test_valid]
    if x_train.shape[0] < 2 or x_test.shape[0] < 1:
        raise ValueError("Not enough valid train/test rows for downstream evaluation.")
    return x_train, y_train, x_test, y_test


def _fit_and_score_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    task: str,
) -> dict[str, float]:
    scores, _, _ = _fit_and_score_models_with_predictions(
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, task=task
    )
    return scores


def _fit_and_score_models_with_predictions(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    task: str,
) -> tuple[dict[str, float], dict[str, np.ndarray | None], dict[str, np.ndarray | None]]:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        r2_score,
        roc_auc_score,
    )

    if task not in {"classification", "regression"}:
        raise ValueError("task must be classification or regression.")

    if task == "classification":
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        linear_model = LogisticRegression(max_iter=1000)
        rf_model = RandomForestClassifier(n_estimators=300, random_state=0)
        linear_fit = linear_model.fit(x_train, y_train)
        rf_fit = rf_model.fit(x_train, y_train)
        linear_pred = linear_fit.predict(x_test)
        rf_pred = rf_fit.predict(x_test)
        linear_score_pred = _binary_score_pred(linear_fit, x_test)
        rf_score_pred = _binary_score_pred(rf_fit, x_test)
        scores: dict[str, float] = {
            "downstream_accuracy_linear": float(accuracy_score(y_test, linear_pred)),
            "downstream_accuracy_random_forest": float(accuracy_score(y_test, rf_pred)),
            "downstream_balanced_accuracy_linear": float(balanced_accuracy_score(y_test, linear_pred)),
            "downstream_balanced_accuracy_random_forest": float(balanced_accuracy_score(y_test, rf_pred)),
            "downstream_f1_linear": float(f1_score(y_test, linear_pred, zero_division=0)),
            "downstream_f1_random_forest": float(f1_score(y_test, rf_pred, zero_division=0)),
        }
        if linear_score_pred is not None and np.unique(y_test).size > 1:
            scores["downstream_roc_auc_linear"] = float(roc_auc_score(y_test, linear_score_pred))
            scores["downstream_average_precision_linear"] = float(average_precision_score(y_test, linear_score_pred))
        if rf_score_pred is not None and np.unique(y_test).size > 1:
            scores["downstream_roc_auc_random_forest"] = float(roc_auc_score(y_test, rf_score_pred))
            scores["downstream_average_precision_random_forest"] = float(average_precision_score(y_test, rf_score_pred))
        preds: dict[str, np.ndarray | None] = {"linear": linear_pred, "random_forest": rf_pred, "xgboost": None}
        score_preds: dict[str, np.ndarray | None] = {
            "linear": linear_score_pred,
            "random_forest": rf_score_pred,
            "xgboost": None,
        }
    else:
        linear_model = LinearRegression()
        rf_model = RandomForestRegressor(n_estimators=300, random_state=0)
        linear_pred = linear_model.fit(x_train, y_train).predict(x_test)
        rf_pred = rf_model.fit(x_train, y_train).predict(x_test)
        scores = {
            "downstream_r2_linear": float(r2_score(y_test, linear_pred)),
            "downstream_r2_random_forest": float(r2_score(y_test, rf_pred)),
        }
        preds = {"linear": linear_pred, "random_forest": rf_pred, "xgboost": None}
        score_preds = {"linear": None, "random_forest": None, "xgboost": None}

    try:
        from xgboost import XGBClassifier, XGBRegressor

        if task == "classification":
            xgb_model = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=0,
                eval_metric="mlogloss",
            )
            xgb_pred = xgb_model.fit(x_train, y_train).predict(x_test)
            scores["downstream_accuracy_xgboost"] = float(accuracy_score(y_test, xgb_pred))
            scores["downstream_balanced_accuracy_xgboost"] = float(balanced_accuracy_score(y_test, xgb_pred))
            scores["downstream_f1_xgboost"] = float(f1_score(y_test, xgb_pred, zero_division=0))
            xgb_score_pred = _binary_score_pred(xgb_model, x_test)
            if xgb_score_pred is not None and np.unique(y_test).size > 1:
                scores["downstream_roc_auc_xgboost"] = float(roc_auc_score(y_test, xgb_score_pred))
                scores["downstream_average_precision_xgboost"] = float(average_precision_score(y_test, xgb_score_pred))
            score_preds["xgboost"] = np.asarray(xgb_score_pred) if xgb_score_pred is not None else None
        else:
            xgb_model = XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=0,
            )
            xgb_pred = xgb_model.fit(x_train, y_train).predict(x_test)
            scores["downstream_r2_xgboost"] = float(r2_score(y_test, xgb_pred))
        preds["xgboost"] = np.asarray(xgb_pred)
    except ImportError:
        # Optional dependency: keep evaluation running for linear/RF.
        pass

    return scores, preds, score_preds


def _pool_predictions(predictions: list[np.ndarray], task: str) -> np.ndarray | None:
    if len(predictions) == 0:
        return None
    stacked = np.stack(predictions, axis=0)
    if task == "classification":
        # Majority vote across imputations.
        pooled = []
        for j in range(stacked.shape[1]):
            vals, counts = np.unique(stacked[:, j].astype(int), return_counts=True)
            pooled.append(int(vals[np.argmax(counts)]))
        return np.array(pooled, dtype=int)
    return np.mean(stacked, axis=0)


def _pool_score_predictions(score_predictions: list[np.ndarray]) -> np.ndarray | None:
    if len(score_predictions) == 0:
        return None
    stacked = np.stack(score_predictions, axis=0)
    return np.mean(stacked, axis=0)


def _binary_score_pred(model: object, x_test: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_test)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
    if hasattr(model, "decision_function"):
        score = model.decision_function(x_test)
        return np.asarray(score, dtype=np.float64)
    return None


def _score_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, task: str, metric_key: str) -> float:
    from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, r2_score, roc_auc_score

    if task == "classification":
        y_true_i = y_true.astype(int)
        if metric_key == "accuracy":
            return float(accuracy_score(y_true_i, y_pred.astype(int)))
        if metric_key == "balanced_accuracy":
            return float(balanced_accuracy_score(y_true_i, y_pred.astype(int)))
        if metric_key == "f1":
            return float(f1_score(y_true_i, y_pred.astype(int), zero_division=0))
        if metric_key == "roc_auc":
            if np.unique(y_true_i).size < 2:
                return float("nan")
            return float(roc_auc_score(y_true_i, y_pred.astype(float)))
        if metric_key == "average_precision":
            if np.unique(y_true_i).size < 2:
                return float("nan")
            return float(average_precision_score(y_true_i, y_pred.astype(float)))
        raise ValueError(f"Unsupported classification metric_key: {metric_key}")
    if metric_key != "r2":
        raise ValueError(f"Unsupported regression metric_key: {metric_key}")
    return float(r2_score(y_true, y_pred))
