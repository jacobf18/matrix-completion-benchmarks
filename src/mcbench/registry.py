from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")


class Registry(dict[str, T]):
    def register(self, name: str) -> Callable[[T], T]:
        if not name:
            raise ValueError("Registry name cannot be empty.")

        def decorator(obj: T) -> T:
            if name in self:
                raise ValueError(f"'{name}' already registered.")
            self[name] = obj
            return obj

        return decorator

