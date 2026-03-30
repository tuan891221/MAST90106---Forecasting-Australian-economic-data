from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    def __init__(self, name: str, max_lag: int = 4):
        self.name = name
        self.max_lag = max_lag

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> None:
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...