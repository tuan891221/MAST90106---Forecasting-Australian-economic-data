from __future__ import annotations

import numpy as np

from src.models.base_model import BaseModel


class ARModel(BaseModel):
    def __init__(self, max_lag: int = 4):
        super().__init__(name="ar", max_lag=max_lag)
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names=None) -> None:
        X1 = np.column_stack([np.ones(len(X)), X])
        self.coef_ = np.linalg.pinv(X1.T @ X1) @ X1.T @ y

    def predict(self, X: np.ndarray) -> np.ndarray:
        X1 = np.column_stack([np.ones(len(X)), X])
        return X1 @ self.coef_