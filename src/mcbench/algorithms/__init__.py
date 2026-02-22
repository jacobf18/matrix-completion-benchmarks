from __future__ import annotations

from ..registry import Registry
from .base import MatrixCompletionAlgorithm

ALGORITHM_REGISTRY: Registry[type[MatrixCompletionAlgorithm]] = Registry()

# Import built-ins for side-effect registration.
from . import baselines  # noqa: F401,E402

