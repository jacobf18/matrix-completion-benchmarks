from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import Mask, Matrix


class MatrixMetric(ABC):
    @abstractmethod
    def compute(self, y_true: Matrix, y_pred: Matrix, eval_mask: Mask, **kwargs: object) -> float:
        """Return scalar metric over eval_mask cells."""

