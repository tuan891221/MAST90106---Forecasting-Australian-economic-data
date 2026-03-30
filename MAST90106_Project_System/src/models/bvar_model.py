from __future__ import annotations

import numpy as np

from src.models.base_model import BaseModel


class BVARModel(BaseModel):
    """
    Minnesota-style ridge approximation:
    stronger shrinkage on other-variable lags than own-variable lags.
    """

    def __init__(self, max_lag: int = 4, lambda_1: float = 0.2, lambda_2: float = 0.5, lambda_3: float = 1.0):
        super().__init__(name="bvar", max_lag=max_lag)
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.coef_: np.ndarray | None = None
        self.feature_names: list[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> None:
        self.feature_names = feature_names or [f"x{i}" for i in range(X.shape[1])]
        X1 = np.column_stack([np.ones(len(X)), X])

        penalties = [0.0]
        target_var = None
        for name in self.feature_names:
            if name.endswith("_lag1"):
                target_var = name.rsplit("_lag", 1)[0]
                break

        for name in self.feature_names:
            lag = 1
            if "_lag" in name:
                try:
                    lag = int(name.split("_lag")[-1])
                except ValueError:
                    lag = 1
            var_name = name.rsplit("_lag", 1)[0] if "_lag" in name else name
            scale = self.lambda_1 / (lag ** self.lambda_3)
            if target_var is not None and var_name != target_var:
                scale = scale / self.lambda_2
            penalties.append(scale)

        penalty_matrix = np.diag(penalties)
        self.coef_ = np.linalg.pinv(X1.T @ X1 + penalty_matrix) @ X1.T @ y

    def predict(self, X: np.ndarray) -> np.ndarray:
        X1 = np.column_stack([np.ones(len(X)), X])
        return X1 @ self.coef_