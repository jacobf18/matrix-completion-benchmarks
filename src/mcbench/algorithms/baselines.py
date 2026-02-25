from __future__ import annotations

import numpy as np

from . import ALGORITHM_REGISTRY
from .base import MatrixCompletionAlgorithm


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
                X = fancy_check_array(X, force_all_finite=False)
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

        plugin_kwargs = dict(kwargs)
        plugin = Imputers().get("hyperimpute", **plugin_kwargs)
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
