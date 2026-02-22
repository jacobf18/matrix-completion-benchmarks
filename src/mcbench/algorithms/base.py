from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import Matrix


class MatrixCompletionAlgorithm(ABC):
    @abstractmethod
    def complete(self, observed: Matrix, **kwargs: object) -> Matrix:
        """Return a dense matrix with missing values filled."""

