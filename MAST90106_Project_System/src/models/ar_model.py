from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base_model import BaseTimeSeriesModel
from src.utils.constants import VARIABLES


class ARModel(BaseTimeSeriesModel):
    """
    Separate ARX model for each endogenous variable:
    y_t = intercept + own lags + optional exogenous COVID dummies
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        max_lag = cfg.get("models", {}).get("ar", {}).get("max_lag", 4)
        super().__init__(config=cfg, max_lag=max_lag)
        self.ridge_alpha = float(cfg.get("models", {}).get("ar", {}).get("ridge_alpha", 1e-4))
        self.coef_: dict[str, np.ndarray] = {}

    def fit(self, data: pd.DataFrame) -> "ARModel":
        self.history_ = data.copy()
        self.endog_cols = [c for c in VARIABLES if c in data.columns]
        self.exog_cols = [c for c in data.columns if c not in self.endog_cols]
        self.coef_ = {}

        clean = data[self.endog_cols + self.exog_cols].copy()
        for col in clean.columns:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")

        for target in self.endog_cols:
            df = clean[[target] + self.exog_cols].copy()

            for lag in range(1, self.max_lag + 1):
                df[f"{target}_lag{lag}"] = df[target].shift(lag)

            df = df.dropna().reset_index(drop=True)
            if len(df) <= self.max_lag + 1:
                continue

            y = df[target].to_numpy(dtype=float)
            X_cols = [f"{target}_lag{lag}" for lag in range(1, self.max_lag + 1)] + self.exog_cols
            X = df[X_cols].to_numpy(dtype=float)
            X1 = np.column_stack([np.ones(len(X)), X])

            penalty = np.eye(X1.shape[1]) * self.ridge_alpha
            penalty[0, 0] = 0.0

            beta = np.linalg.solve(X1.T @ X1 + penalty, X1.T @ y)
            self.coef_[target] = beta

        return self

    def forecast(self, h: int, future_exog: pd.DataFrame | None = None) -> pd.DataFrame:
        if self.history_ is None:
            raise ValueError("Model must be fitted before forecasting.")

        hist = self.history_.copy()

        if self.exog_cols:
            if future_exog is None:
                last_exog = hist[self.exog_cols].iloc[[-1]].copy()
                future_exog = pd.concat([last_exog] * h, ignore_index=True)
            else:
                future_exog = future_exog[self.exog_cols].copy().reset_index(drop=True)
        else:
            future_exog = pd.DataFrame(index=range(h))

        out_rows = []

        for step in range(h):
            next_row = {}

            for target in self.endog_cols:
                if target not in self.coef_:
                    next_row[target] = float(hist[target].iloc[-1])
                    continue

                beta = self.coef_[target]
                lags = [float(hist[target].iloc[-lag]) for lag in range(1, self.max_lag + 1)]

                exog_vals = future_exog.iloc[step].tolist() if self.exog_cols else []
                x = np.array([1.0] + lags + exog_vals, dtype=float)
                next_row[target] = float(x @ beta)

            # keep exog in history only for recursive stepping
            for col in self.exog_cols:
                next_row[col] = float(future_exog.iloc[step][col])

            out_rows.append({k: next_row[k] for k in self.endog_cols})
            hist = pd.concat([hist, pd.DataFrame([next_row])], ignore_index=True)

        return pd.DataFrame(out_rows, columns=self.endog_cols)