from __future__ import annotations

import numpy as np


def singular_value_thresholding(
    observed: np.ndarray,
    tau: float | None = None,
    delta: float | None = None,
    eps: float = 1e-2,
    max_iter: int = 1000,
    iter_print: int = 0,
) -> np.ndarray:
    """Matrix completion via singular value thresholding.

    Adapted from the public GitHub gist by Jacob Reinhold:
    https://gist.github.com/jcreinhold/f026ecfcd0e8a8b80427707aab182e0c
    """

    data = np.asarray(observed, dtype=np.float64)
    mask = np.isfinite(data)
    if not np.any(mask):
        raise ValueError("Input has no finite entries for SVT.")

    x = np.where(mask, data, 0.0)
    z = np.zeros_like(x)

    tau = float(tau) if tau is not None else 2.5 * float(np.sum(x.shape))
    observed_count = int(np.sum(mask))
    delta = float(delta) if delta is not None else 1.2 * float(np.prod(x.shape)) / float(observed_count)

    denom = float(np.linalg.norm(mask * x))
    if denom == 0.0:
        # All observed entries are zero. The zero matrix is already a valid completion.
        return np.zeros_like(x)

    for i in range(max_iter):
        u, s, vt = np.linalg.svd(z, full_matrices=False)
        s = np.maximum(s - tau, 0.0)
        a = u @ np.diag(s) @ vt
        z += delta * mask * (x - a)
        error = float(np.linalg.norm(mask * (x - a)) / denom)
        if iter_print and i % iter_print == 0:
            print(f"Iteration: {i}; Error: {error:.4e}")
        if error < eps:
            return a

    return a
