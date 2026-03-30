from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.models.base_model import BaseModel


class FactorModel(BaseModel):
    def __init__(self, max_lag: int = 4, n_factors: int = 2):
        super().__init__(name="factor", max_lag=max_lag)
        self.n_factors = n_factors
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_factors)
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names=None) -> None:
        Xs = self.scaler.fit_transform(X)
        F = self.pca.fit_transform(Xs)
        F1 = np.column_stack([np.ones(len(F)), F])
        self.coef_ = np.linalg.pinv(F1.T @ F1) @ F1.T @ y

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        F = self.pca.transform(Xs)
        F1 = np.column_stack([np.ones(len(F)), F])
        return F1 @ self.coef_