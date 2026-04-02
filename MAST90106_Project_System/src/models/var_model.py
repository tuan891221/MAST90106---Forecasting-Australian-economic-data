from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base_model import BaseTimeSeriesModel
from src.utils.constants import VARIABLES


def make_varx_lagged_data(
    data: pd.DataFrame,
    endog_cols: list[str],
    exog_cols: list[str],
    p: int,
):
    values = data[endog_cols].to_numpy(dtype=float)
    T, n = values.shape

    Y = values[p:]
    X_parts = [np.ones((T - p, 1))]

    for lag in range(1, p + 1):
        X_parts.append(values[p - lag : T - lag])

    if exog_cols:
        exog_vals = data[exog_cols].to_numpy(dtype=float)
        X_parts.append(exog_vals[p:])

    X = np.hstack(X_parts)
    return Y, X


class VARModel(BaseTimeSeriesModel):
    """
    VARX-style ridge regression:
    all endogenous variables jointly, optional exogenous COVID dummies.
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        max_lag = cfg.get("models", {}).get("var", {}).get("max_lag", 4)
        super().__init__(config=cfg, max_lag=max_lag)
        self.ridge_alpha = float(cfg.get("models", {}).get("var", {}).get("ridge_alpha", 1e-3))
        self.B_: np.ndarray | None = None

    def fit(self, data: pd.DataFrame) -> "VARModel":
        self.history_ = data.copy()
        self.endog_cols = [c for c in VARIABLES if c in data.columns]
        self.exog_cols = [c for c in data.columns if c not in self.endog_cols]

        clean = data[self.endog_cols + self.exog_cols].copy()
        for col in clean.columns:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")
        clean = clean.dropna().reset_index(drop=True)

        if len(clean) <= self.max_lag + 1:
            raise ValueError("Not enough data to fit VAR.")

        Y, X = make_varx_lagged_data(clean, self.endog_cols, self.exog_cols, self.max_lag)

        penalty = np.eye(X.shape[1]) * self.ridge_alpha
        penalty[0, 0] = 0.0

        self.B_ = np.linalg.solve(X.T @ X + penalty, X.T @ Y)
        return self

    def forecast(self, h: int, future_exog: pd.DataFrame | None = None) -> pd.DataFrame:
        if self.history_ is None or self.B_ is None:
            raise ValueError("Model must be fitted before forecasting.")

        hist = self.history_.copy()
        endog_values = hist[self.endog_cols].to_numpy(dtype=float).copy()

        if self.exog_cols:
            if future_exog is None:
                last_exog = hist[self.exog_cols].iloc[[-1]].copy()
                future_exog = pd.concat([last_exog] * h, ignore_index=True)
            else:
                future_exog = future_exog[self.exog_cols].copy().reset_index(drop=True)
        else:
            future_exog = pd.DataFrame(index=range(h))

        forecasts = []

        for step in range(h):
            x = [1.0]
            for lag in range(1, self.max_lag + 1):
                x.extend(endog_values[-lag, :])

            if self.exog_cols:
                x.extend(future_exog.iloc[step].to_numpy(dtype=float).tolist())

            x = np.array(x, dtype=float)
            y_hat = x @ self.B_
            forecasts.append(y_hat)
            endog_values = np.vstack([endog_values, y_hat])

        return pd.DataFrame(forecasts, columns=self.endog_cols)