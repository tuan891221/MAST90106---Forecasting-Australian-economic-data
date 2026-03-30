from __future__ import annotations

import numpy as np

from src.models.base_model import BaseModel


class MeanModel(BaseModel):
    def __init__(self):
        super().__init__(name="mean", max_lag=1)
        self.mean_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names=None) -> None:
        self.mean_ = float(np.nanmean(y))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(shape=(X.shape[0],), fill_value=self.mean_)