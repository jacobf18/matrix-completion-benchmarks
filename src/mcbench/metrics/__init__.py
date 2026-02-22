from __future__ import annotations

from ..registry import Registry
from .base import MatrixMetric

METRIC_REGISTRY: Registry[type[MatrixMetric]] = Registry()

# Import built-ins for side-effect registration.
from . import common  # noqa: F401,E402

