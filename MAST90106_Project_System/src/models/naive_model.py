from __future__ import annotations

import numpy as np

from src.models.base_model import BaseModel


class NaiveModel(BaseModel):
    def __init__(self):
        super().__init__(name="naive", max_lag=1)

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names=None) -> None:
        return

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X[:, 0]