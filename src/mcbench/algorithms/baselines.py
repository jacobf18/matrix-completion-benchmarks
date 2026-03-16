from __future__ import annotations

import json
import numpy as np
from pathlib import Path

from . import ALGORITHM_REGISTRY
from .base import MatrixCompletionAlgorithm
from .external_svt import singular_value_thresholding
import warnings


def _wrap_check_array_force_all_finite(check_array_fn: object) -> object:
    try:
        param_names = check_array_fn.__code__.co_varnames  # type: ignore[attr-defined]
    except AttributeError:
        return check_array_fn

    if "force_all_finite" in param_names or "ensure_all_finite" not in param_names:
        return check_array_fn

    def _check_array_compat(*args: object, **kwargs: object) -> np.ndarray:
        if "force_all_finite" in kwargs and "ensure_all_finite" not in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        else:
            kwargs.pop("force_all_finite", None)
        return check_array_fn(*args, **kwargs)

    return _check_array_compat


def _patch_fancyimpute_sklearn_compat() -> None:
    """Allow fancyimpute to call sklearn.check_array(force_all_finite=...)."""
    try:
        import sklearn.utils.validation as sk_validation
    except ImportError:
        return

    check_array = getattr(sk_validation, "check_array", None)
    if check_array is None:
        return

    patched = _wrap_check_array_force_all_finite(check_array)
    sk_validation.check_array = patched  # type: ignore[assignment]

    try:
        import importlib
    except ImportError:
        return

    module_names = (
        "fancyimpute.solver",
        "fancyimpute.soft_impute",
        "fancyimpute.nuclear_norm_minimization",
    )
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        if getattr(module, "check_array", None) is not None:
            module.check_array = _wrap_check_array_force_all_finite(module.check_array)  # type: ignore[assignment]


def _prefill_missing_with_column_means(observed: np.ndarray) -> np.ndarray:
    data = observed.astype(np.float64, copy=True)
    missing = ~np.isfinite(data)
    if not np.any(missing):
        return data

    finite = np.isfinite(data)
    if not np.any(finite):
        raise ValueError("Input has no finite entries to estimate fill values.")

    global_mean = float(np.nanmean(data))
    col_means = np.nanmean(data, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, global_mean)
    missing_rows, missing_cols = np.where(missing)
    data[missing_rows, missing_cols] = col_means[missing_cols]
    return data


@ALGORITHM_REGISTRY.register("global_mean")
class GlobalMeanImputer(MatrixCompletionAlgorithm):
    def complete(self, observed: np.ndarray, **kwargs: object) -> np.ndarray:
        out = observed.astype(np.float64, copy=True)
        finite = np.isfinite(out)
        if not np.any(finite):
            raise ValueError("Input has no finite entries to estimate a fill value.")
        global_mean = float(np.nanmean(out))
        out[~finite] = global_mean
        return out


@ALGORITHM_REGISTRY.register("row_mean")
class RowMeanImputer(MatrixCompletionAlgorithm):
    def complete(self, observed: np.ndarray, **kwargs: object) -> np.ndarray:
        out = observed.astype(np.float64, copy=True)
        finite = np.isfinite(out)
        if not np.any(finite):
            raise ValueError("Input has no finite entries to estimate fill values.")
        global_mean = float(np.nanmean(out))
        row_means = np.nanmean(out, axis=1)
        row_means = np.where(np.isfinite(row_means), row_means, global_mean)
        missing_rows, missing_cols = np.where(~finite)
        out[missing_rows, missing_cols] = row_means[missing_rows]
        return out


@ALGORITHM_REGISTRY.register("col_mean")
class ColumnMeanImputer(MatrixCompletionAlgorithm):
    def complete(self, observed: np.ndarray, **kwargs: object) -> np.ndarray:
        out = observed.astype(np.float64, copy=True)
        finite = np.isfinite(out)
        if not np.any(finite):
            raise ValueError("Input has no finite entries to estimate fill values.")
        global_mean = float(np.nanmean(out))
        col_means = np.nanmean(out, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, global_mean)
        missing_rows, missing_cols = np.where(~finite)
        out[missing_rows, missing_cols] = col_means[missing_cols]
        return out


@ALGORITHM_REGISTRY.register("col_mode")
class ColumnModeImputer(MatrixCompletionAlgorithm):
    def complete(self, observed: np.ndarray, **kwargs: object) -> np.ndarray:
        out = observed.astype(np.float64, copy=True)
        finite = np.isfinite(out)
        if not np.any(finite):
            raise ValueError("Input has no finite entries to estimate fill values.")

        global_mean = float(np.nanmean(out))
        fill_values = np.full(out.shape[1], global_mean, dtype=np.float64)

        for col in range(out.shape[1]):
            vals = out[finite[:, col], col]
            if vals.size == 0:
                continue
            unique_vals, counts = np.unique(vals, return_counts=True)
            mode_candidates = unique_vals[counts == np.max(counts)]
            fill_values[col] = float(np.min(mode_candidates))

        missing_rows, missing_cols = np.where(~finite)
        out[missing_rows, missing_cols] = fill_values[missing_cols]
        return out


@ALGORITHM_REGISTRY.register("svt")
class SingularValueThresholding(MatrixCompletionAlgorithm):
    def complete(self, observed: np.ndarray, **kwargs: object) -> np.ndarray:
        return singular_value_thresholding(
            observed=observed,
            tau=kwargs.get("tau"),
            delta=kwargs.get("delta"),
            eps=float(kwargs.get("eps", 1e-2)),
            max_iter=int(kwargs.get("max_iters", 1000)),
            iter_print=int(kwargs.get("iter_print", 0)),
        )


@ALGORITHM_REGISTRY.register("knn")
class KNNImputerBaseline(MatrixCompletionAlgorithm):
    def complete(self, observed: np.ndarray, **kwargs: object) -> np.ndarray:
        try:
            from sklearn.impute import KNNImputer
        except ImportError as exc:
            raise ImportError(
                "knn requires optional dependency 'scikit-learn'. "
                "Install with: pip install scikit-learn"
            ) from exc

        data = observed.astype(np.float64, copy=True)
        if not np.any(~np.isfinite(data)):
            return data

        # KNNImputer drops columns that are entirely missing; prefill to preserve shape.
        finite_per_col = np.any(np.isfinite(data), axis=0)
        if not np.all(finite_per_col):
            finite_counts = np.sum(np.isfinite(data), axis=0)
            finite_sums = np.nansum(data, axis=0)
            col_means = np.divide(
                finite_sums,
                finite_counts,
                out=np.full(data.shape[1], np.nan, dtype=np.float64),
                where=finite_counts > 0,
            )
            global_mean = float(np.nanmean(data))
            col_means = np.where(np.isfinite(col_means), col_means, global_mean)
            missing_cols = np.flatnonzero(~finite_per_col)
            for col in missing_cols:
                data[:, col] = col_means[col]

        imputer = KNNImputer(
            n_neighbors=int(kwargs.get("n_neighbors", 5)),
            weights=str(kwargs.get("weights", "uniform")),
            metric=str(kwargs.get("metric", "nan_euclidean")),
        )
        return np.asarray(imputer.fit_transform(data), dtype=np.float64)


@ALGORITHM_REGISTRY.register("soft_impute")
class SoftImpute(MatrixCompletionAlgorithm):
    def complete(self, observed: np.ndarray, **kwargs: object) -> np.ndarray:
        _patch_fancyimpute_sklearn_compat()
        try:
            from fancyimpute import SoftImpute as FancySoftImpute
        except ImportError as exc:
            raise ImportError(
                "soft_impute requires the optional dependency 'fancyimpute'. "
                "Install with: pip install fancyimpute"
            ) from exc

        solver = FancySoftImpute(
            shrinkage_value=kwargs.get("shrinkage"),
            max_rank=kwargs.get("rank"),
            max_iters=int(kwargs.get("max_iters", 100)),
            convergence_threshold=float(kwargs.get("tol", 1e-5)),
            init_fill_method=str(kwargs.get("init_fill", "zero")),
            verbose=bool(kwargs.get("verbose", False)),
        )
        return np.asarray(solver.fit_transform(observed.astype(np.float64, copy=True)), dtype=np.float64)


@ALGORITHM_REGISTRY.register("nuclear_norm_minimization")
class NuclearNormMinimization(MatrixCompletionAlgorithm):
    def complete(self, observed: np.ndarray, **kwargs: object) -> np.ndarray:
        _patch_fancyimpute_sklearn_compat()
        try:
            from fancyimpute import NuclearNormMinimization as FancyNNM
            from fancyimpute.nuclear_norm_minimization import check_array as fancy_check_array
        except ImportError as exc:
            raise ImportError(
                "nuclear_norm_minimization requires the optional dependency 'fancyimpute'. "
                "Install with: pip install fancyimpute"
            ) from exc

        backend = str(kwargs.pop("cvx_solver", "SCS")).upper()

        solver = FancyNNM(
            require_symmetric_solution=bool(kwargs.get("require_symmetric_solution", False)),
            min_value=kwargs.get("min_value"),
            max_value=kwargs.get("max_value"),
            error_tolerance=float(kwargs.get("error_tolerance", 1e-8)),
            max_iters=int(kwargs.get("max_iters", 50000)),
            verbose=bool(kwargs.get("verbose", False)),
        )

        if backend != "CVXOPT":
            try:
                import cvxpy
            except ImportError as exc:
                raise ImportError(
                    "nuclear_norm_minimization with non-CVXOPT solvers requires cvxpy. "
                    "Install with: pip install cvxpy"
                ) from exc

            def _solve_with_backend(self: object, X: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
                X = fancy_check_array(X, ensure_all_finite=False)
                m, n = X.shape
                S, objective = self._create_objective(m, n)  # type: ignore[attr-defined]
                constraints = self._constraints(  # type: ignore[attr-defined]
                    X=X,
                    missing_mask=missing_mask,
                    S=S,
                    error_tolerance=self.error_tolerance,  # type: ignore[attr-defined]
                )
                problem = cvxpy.Problem(objective, constraints)
                problem.solve(
                    verbose=self.verbose,  # type: ignore[attr-defined]
                    solver=getattr(cvxpy, backend),
                    max_iters=self.max_iters,  # type: ignore[attr-defined]
                )
                return S.value

            solver.solve = _solve_with_backend.__get__(solver, solver.__class__)  # type: ignore[assignment]

        return np.asarray(solver.fit_transform(observed.astype(np.float64, copy=True)), dtype=np.float64)


@ALGORITHM_REGISTRY.register("robust_pca")
class RobustPCABaseline(MatrixCompletionAlgorithm):
    def complete(self, observed: np.ndarray, **kwargs: object) -> np.ndarray:
        try:
            from rpca import RobustPCA
        except ImportError as exc:
            raise ImportError(
                "robust_pca requires optional dependency 'rpca'. "
                "Install with: pip install rpca"
            ) from exc

        data = _prefill_missing_with_column_means(observed)
        model = RobustPCA(
            n_components=kwargs.get("n_components"),
            max_iter=int(kwargs.get("max_iters", 100)),
            tol=float(kwargs.get("tol", 1e-5)),
            beta=kwargs.get("beta"),
            beta_init=kwargs.get("beta_init"),
            gamma=float(kwargs.get("gamma", 0.5)),
            mu=kwargs.get("mu", (5, 5)),
            trim=bool(kwargs.get("trim", False)),
            verbose=bool(kwargs.get("verbose", False)),
            copy=bool(kwargs.get("copy", True)),
        )
        model.fit(data)
        low_rank = np.asarray(model.low_rank_, dtype=np.float64)
        mean = np.asarray(model.mean_, dtype=np.float64)
        if mean.ndim != 1 or mean.shape[0] != low_rank.shape[1]:
            raise ValueError("robust_pca returned an unexpected mean vector shape.")
        low_rank = low_rank + mean[None, :]

        if low_rank.shape != data.shape:
            raise ValueError(
                f"robust_pca returned shape {low_rank.shape}, expected {data.shape}."
            )
        return low_rank


@ALGORITHM_REGISTRY.register("hyperimpute")
class HyperImputeBaseline(MatrixCompletionAlgorithm):
    def complete(self, observed: np.ndarray, **kwargs: object) -> np.ndarray:
        try:
            import pandas as pd
            from hyperimpute.plugins.imputers import Imputers
        except ImportError as exc:
            raise ImportError(
                "hyperimpute requires optional dependencies 'hyperimpute' and 'pandas'. "
                "Install with: pip install hyperimpute pandas"
            ) from exc

        data = observed.astype(np.float64, copy=True)
        if not np.any(~np.isfinite(data)):
            return data

        # HyperImpute plugins can fail when a column is entirely missing.
        finite_per_col = np.any(np.isfinite(data), axis=0)
        if not np.all(finite_per_col):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                col_means = np.nanmean(data, axis=0)
            col_means = np.where(np.isfinite(col_means), col_means, 0.0)
            missing_cols = np.flatnonzero(~finite_per_col)
            for col in missing_cols:
                data[:, col] = col_means[col]

        plugin_kwargs = dict(kwargs)
        plugin = Imputers().get("hyperimpute", **plugin_kwargs)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            transformed = plugin.fit_transform(pd.DataFrame(data))
            return np.asarray(transformed, dtype=np.float64)


@ALGORITHM_REGISTRY.register("missforest")
class MissForestBaseline(MatrixCompletionAlgorithm):
    def complete(self, observed: np.ndarray, **kwargs: object) -> np.ndarray:
        try:
            import pandas as pd
            from hyperimpute.plugins.imputers import Imputers
        except ImportError as exc:
            raise ImportError(
                "missforest requires optional dependencies 'hyperimpute' and 'pandas'. "
                "Install with: pip install hyperimpute pandas"
            ) from exc

        data = observed.astype(np.float64, copy=True)
        if not np.any(~np.isfinite(data)):
            return data

        # MissForest can fail when a column is entirely missing.
        finite_per_col = np.any(np.isfinite(data), axis=0)
        if not np.all(finite_per_col):
            col_means = np.nanmean(data, axis=0)
            col_means = np.where(np.isfinite(col_means), col_means, 0.0)
            missing_cols = np.flatnonzero(~finite_per_col)
            for col in missing_cols:
                data[:, col] = col_means[col]

        plugin_kwargs = dict(kwargs)
        plugin = Imputers().get("missforest", **plugin_kwargs)
        transformed = plugin.fit_transform(pd.DataFrame(data))
        return np.asarray(transformed, dtype=np.float64)


@ALGORITHM_REGISTRY.register("forest_diffusion")
class ForestDiffusionBaseline(MatrixCompletionAlgorithm):
    def complete(self, observed: np.ndarray, **kwargs: object) -> np.ndarray:
        try:
            from ForestDiffusion import ForestDiffusionModel
        except ImportError as exc:
            raise ImportError(
                "forest_diffusion requires optional dependency 'ForestDiffusion'. "
                "Install with: pip install ForestDiffusion"
            ) from exc

        data = observed.astype(np.float64, copy=True)
        if not np.any(~np.isfinite(data)):
            return data

        model_kwargs = dict(kwargs)
        k = int(model_kwargs.pop("k", 1))
        model = ForestDiffusionModel(X=data, label_y=None, **model_kwargs)
        imputed = model.impute(k=k)
        return np.asarray(imputed, dtype=np.float64)


@ALGORITHM_REGISTRY.register("tab_impute")
class TabImputePFNBaseline(MatrixCompletionAlgorithm):
    def complete(self, observed: np.ndarray, **kwargs: object) -> np.ndarray:
        try:
            from tabimpute.interface import ImputePFN
        except ImportError as exc:
            raise ImportError(
                "tab_impute requires optional dependency 'tabimpute'. "
                "Install with: pip install tabimpute"
            ) from exc

        data = observed.astype(np.float64, copy=True)
        if not np.any(~np.isfinite(data)):
            return data

        device = str(kwargs.get("device", "cuda"))
        nhead = int(kwargs.get("nhead", 2))
        checkpoint_path = kwargs.get("checkpoint_path")
        max_num_rows = kwargs.get("max_num_rows")
        max_num_chunks = kwargs.get("max_num_chunks")
        verbose = bool(kwargs.get("verbose", False))
        num_repeats = int(kwargs.get("num_repeats", 1))
        model_version = str(kwargs.get("model_version", "2")).strip().lower()

        model: object
        if model_version in {"2", "v2"}:
            checkpoint_path_v2 = kwargs.get("v2_checkpoint_path", checkpoint_path)
            try:
                from tabimpute.tabimpute_v2 import TabImputeV2
            except ImportError as exc:
                raise ImportError(
                    "tab_impute model_version=2 requires tabimpute.tabimpute_v2.TabImputeV2."
                ) from exc
            model = TabImputeV2(
                device=device,
                nhead=nhead,
                checkpoint_path=checkpoint_path_v2,
                max_num_rows=max_num_rows,
                max_num_chunks=max_num_chunks,
                verbose=verbose,
            )
        elif model_version in {"1", "v1"}:
            model = ImputePFN(
                device=device,
                nhead=nhead,
                checkpoint_path=checkpoint_path,
                max_num_rows=max_num_rows,
                max_num_chunks=max_num_chunks,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Unknown tab_impute model_version '{model_version}'. Use 1 or 2.")

        output = model.impute(data, return_full=False, num_repeats=num_repeats)

        pred = np.asarray(output, dtype=np.float64)
        if num_repeats > 1 and pred.ndim == 3:
            pred = np.mean(pred, axis=0)
        if pred.shape != data.shape:
            raise ValueError(
                f"tab_impute returned shape {pred.shape}, expected {data.shape}."
            )
        return pred


@ALGORITHM_REGISTRY.register("tab_impute_constraints")
@ALGORITHM_REGISTRY.register("TabImputeConstraints")
class TabImputeConstraintsBaseline(MatrixCompletionAlgorithm):
    def complete(self, observed: np.ndarray, **kwargs: object) -> np.ndarray:
        try:
            import torch
            from tabimpute.postprocessing.bounds import apply_bounds_constraint
            from tabimpute.tabimpute_v2 import TabImputeV2
        except ImportError as exc:
            raise ImportError(
                "tab_impute_constraints requires optional dependency 'tabimpute' with v2 and bounds postprocessing."
            ) from exc

        data = observed.astype(np.float64, copy=True)
        if not np.any(~np.isfinite(data)):
            return data

        constraints_json = kwargs.get("constraints_json")
        dataset_root = kwargs.get("dataset_root")
        target_column = str(kwargs.get("target_column", "EventCKD35"))
        column_names_arg = kwargs.get("column_names")
        if not constraints_json:
            raise ValueError("tab_impute_constraints requires 'constraints_json' in algorithm params.")

        constraints_payload = json.loads(Path(str(constraints_json)).read_text())
        constraints = constraints_payload.get("constraints", {})
        if not isinstance(constraints, dict):
            raise ValueError("constraints_json must include an object under key 'constraints'.")

        if isinstance(column_names_arg, (list, tuple)):
            col_names = [str(c) for c in column_names_arg]
        else:
            if not dataset_root:
                raise ValueError(
                    "tab_impute_constraints requires either 'column_names' or 'dataset_root' in algorithm params."
                )
            source_meta = json.loads((Path(str(dataset_root)) / "source_meta.json").read_text())
            features = [str(c) for c in source_meta.get("feature_columns", [])]
            col_names = [*features, target_column]

        if len(col_names) != data.shape[1]:
            raise ValueError(
                f"column_names length {len(col_names)} does not match observed columns {data.shape[1]}."
            )

        means = np.nanmean(data, axis=0)
        stds = np.nanstd(data, axis=0)
        stds = np.where(np.isnan(stds), 1.0, stds)
        stds = np.where(stds == 0, 1.0, stds)
        means = np.where(np.isnan(means), 0.0, means)
        x_norm = (data - means) / (stds + 1e-16)

        lower_norm = np.full(data.shape[1], np.nan, dtype=np.float32)
        upper_norm = np.full(data.shape[1], np.nan, dtype=np.float32)
        constrained_cols = 0
        for idx, name in enumerate(col_names):
            spec = constraints.get(name)
            if not isinstance(spec, dict):
                continue
            lo = spec.get("lower")
            hi = spec.get("upper")
            if lo is not None:
                lower_norm[idx] = (float(lo) - means[idx]) / (stds[idx] + 1e-16)
            if hi is not None:
                upper_norm[idx] = (float(hi) - means[idx]) / (stds[idx] + 1e-16)
            constrained_cols += 1
        if constrained_cols == 0:
            raise ValueError("No column constraints were matched to the provided column schema.")

        device = str(kwargs.get("device", "cuda"))
        nhead = int(kwargs.get("nhead", 2))
        checkpoint_path = kwargs.get("checkpoint_path")
        max_num_rows = kwargs.get("max_num_rows")
        max_num_chunks = kwargs.get("max_num_chunks")
        verbose = bool(kwargs.get("verbose", False))
        model_version = str(kwargs.get("model_version", "2")).strip().lower()
        if model_version not in {"2", "v2"}:
            raise ValueError("tab_impute_constraints currently supports only model_version=2.")
        checkpoint_path_v2 = kwargs.get("v2_checkpoint_path", checkpoint_path)

        lower_t = torch.tensor(lower_norm, dtype=torch.float32, device=device).view(1, 1, -1)
        upper_t = torch.tensor(upper_norm, dtype=torch.float32, device=device).view(1, 1, -1)

        model = TabImputeV2(
            device=device,
            nhead=nhead,
            checkpoint_path=checkpoint_path_v2,
            max_num_rows=max_num_rows,
            max_num_chunks=max_num_chunks,
            verbose=verbose,
            postprocessor=apply_bounds_constraint,
            postprocessor_kwargs={"lower_bound": lower_t, "upper_bound": upper_t},
        )

        x_imputed_norm, _ = model.get_imputation(x_norm, num_repeats=1)
        pred = np.asarray(x_imputed_norm, dtype=np.float64) * (stds + 1e-16) + means
        if pred.shape != data.shape:
            raise ValueError(f"tab_impute_constraints returned shape {pred.shape}, expected {data.shape}.")
        return pred
