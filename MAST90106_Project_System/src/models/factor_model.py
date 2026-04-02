from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base_model import BaseTimeSeriesModel
from src.utils.constants import VARIABLES


class FactorModel(BaseTimeSeriesModel):
    """
    Factor model with optional exogenous COVID dummies.

    Steps:
    1. standardize endogenous variables
    2. extract principal components
    3. fit VARX on factors
    4. recursively forecast factors
    5. map back to original variable space
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        max_lag = cfg.get("models", {}).get("factor", {}).get("max_lag", 4)
        super().__init__(config=cfg, max_lag=max_lag)

        self.n_factors = int(cfg.get("models", {}).get("factor", {}).get("n_factors", 2))
        self.ridge_alpha = float(cfg.get("models", {}).get("factor", {}).get("ridge_alpha", 1e-3))

        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.loadings_: np.ndarray | None = None
        self.factor_coef_: np.ndarray | None = None
        self.factor_history_: np.ndarray | None = None

    def fit(self, data: pd.DataFrame) -> "FactorModel":
        self.history_ = data.copy()
        self.endog_cols = [c for c in VARIABLES if c in data.columns]
        self.exog_cols = [c for c in data.columns if c not in self.endog_cols]

        clean = data[self.endog_cols + self.exog_cols].copy()
        for col in clean.columns:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")
        clean = clean.dropna().reset_index(drop=True)

        X = clean[self.endog_cols].to_numpy(dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0

        Z = (X - self.mean_) / self.std_

        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        k = min(self.n_factors, Z.shape[1], Vt.shape[0])

        self.loadings_ = Vt[:k].T
        F = Z @ self.loadings_
        self.factor_history_ = F.copy()

        p = self.max_lag
        Y = F[p:]
        X_parts = [np.ones((len(F) - p, 1))]
        for lag in range(1, p + 1):
            X_parts.append(F[p - lag : len(F) - lag])

        if self.exog_cols:
            exog_vals = clean[self.exog_cols].to_numpy(dtype=float)
            X_parts.append(exog_vals[p:])

        Xf = np.hstack(X_parts)

        penalty = np.eye(Xf.shape[1]) * self.ridge_alpha
        penalty[0, 0] = 0.0

        self.factor_coef_ = np.linalg.solve(Xf.T @ Xf + penalty, Xf.T @ Y)
        return self

    def forecast(self, h: int, future_exog: pd.DataFrame | None = None) -> pd.DataFrame:
        if (
            self.factor_coef_ is None
            or self.factor_history_ is None
            or self.loadings_ is None
            or self.mean_ is None
            or self.std_ is None
        ):
            raise ValueError("Model must be fitted before forecasting.")

        if self.exog_cols:
            if future_exog is None:
                last_exog = self.history_[self.exog_cols].iloc[[-1]].copy()
                future_exog = pd.concat([last_exog] * h, ignore_index=True)
            else:
                future_exog = future_exog[self.exog_cols].copy().reset_index(drop=True)
        else:
            future_exog = pd.DataFrame(index=range(h))

        history = self.factor_history_.copy()
        forecasts = []

        for step in range(h):
            x = [1.0]
            for lag in range(1, self.max_lag + 1):
                x.extend(history[-lag, :])

            if self.exog_cols:
                x.extend(future_exog.iloc[step].to_numpy(dtype=float).tolist())

            x = np.array(x, dtype=float)
            f_hat = x @ self.factor_coef_
            history = np.vstack([history, f_hat])

            z_hat = f_hat @ self.loadings_.T
            y_hat = z_hat * self.std_ + self.mean_
            forecasts.append(y_hat)

        return pd.DataFrame(forecasts, columns=self.endog_cols)