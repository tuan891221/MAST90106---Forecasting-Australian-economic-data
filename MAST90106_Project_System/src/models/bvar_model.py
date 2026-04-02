from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.linalg import inv

from src.models.base_model import BaseTimeSeriesModel
from src.utils.constants import VARIABLES


def make_bvar_lagged_data(
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


def estimate_ar_residual_variances(data: pd.DataFrame, endog_cols: list[str], p: int) -> np.ndarray:
    sigmas2 = np.zeros(len(endog_cols))

    for i, col in enumerate(endog_cols):
        y = pd.to_numeric(data[col], errors="coerce").to_numpy(dtype=float)
        Y = y[p:]
        X_parts = [np.ones((len(y) - p, 1))]
        for lag in range(1, p + 1):
            X_parts.append(y[p - lag : len(y) - lag].reshape(-1, 1))
        X = np.hstack(X_parts)

        beta_ols = inv(X.T @ X) @ (X.T @ Y)
        resid = Y - X @ beta_ols
        ddof = min(X.shape[1], max(len(Y) - 1, 1))
        sigmas2[i] = np.var(resid, ddof=ddof)

        if not np.isfinite(sigmas2[i]) or sigmas2[i] <= 0:
            sigmas2[i] = 1.0

    return sigmas2


def minnesota_prior_one_equation(
    n: int,
    p: int,
    n_exog: int,
    eq_i: int,
    sigmas2: np.ndarray,
    lam1: float = 0.2,
    lam2: float = 0.5,
    lam3: float = 100.0,
    exog_var: float = 10.0,
    prior_mean_own_lag1: float = 0.0,
):
    k = 1 + n * p + n_exog
    b0 = np.zeros(k)
    V0 = np.zeros((k, k))

    V0[0, 0] = lam3**2

    idx = 1
    for lag in range(1, p + 1):
        for j in range(n):
            if j == eq_i:
                prior_var = (lam1 / lag) ** 2
            else:
                prior_var = ((lam1 * lam2) / lag) ** 2 * (sigmas2[eq_i] / sigmas2[j])

            V0[idx, idx] = prior_var
            idx += 1

    for _ in range(n_exog):
        V0[idx, idx] = exog_var**2
        idx += 1

    own_first_lag_pos = 1 + eq_i
    b0[own_first_lag_pos] = prior_mean_own_lag1

    return b0, V0


class BVARModel(BaseTimeSeriesModel):
    """
    Multivariate BVAR with Minnesota prior + optional exogenous COVID dummies.
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        max_lag = cfg.get("models", {}).get("bvar", {}).get("max_lag", 4)
        super().__init__(config=cfg, max_lag=max_lag)

        # accept both naming styles
        self.lam1 = float(cfg.get("models", {}).get("bvar", {}).get("lam1",
                          cfg.get("models", {}).get("bvar", {}).get("lambda_1", 0.2)))
        self.lam2 = float(cfg.get("models", {}).get("bvar", {}).get("lam2",
                          cfg.get("models", {}).get("bvar", {}).get("lambda_2", 0.5)))
        self.lam3 = float(cfg.get("models", {}).get("bvar", {}).get("lam3",
                          cfg.get("models", {}).get("bvar", {}).get("lambda_3", 100.0)))
        self.exog_var = float(cfg.get("models", {}).get("bvar", {}).get("exog_var", 10.0))

        self.B_post: np.ndarray | None = None
        self.sigmas2_: np.ndarray | None = None

    def fit(self, data: pd.DataFrame) -> "BVARModel":
        self.history_ = data.copy()
        self.endog_cols = [c for c in VARIABLES if c in data.columns]
        self.exog_cols = [c for c in data.columns if c not in self.endog_cols]

        if not self.endog_cols:
            raise ValueError("No endogenous variables found for BVAR.")

        clean = data[self.endog_cols + self.exog_cols].copy()
        for col in clean.columns:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")
        clean = clean.dropna().reset_index(drop=True)

        if len(clean) <= self.max_lag + 1:
            raise ValueError("Not enough data to fit BVAR.")

        Y, X = make_bvar_lagged_data(clean, self.endog_cols, self.exog_cols, self.max_lag)

        n = len(self.endog_cols)
        n_exog = len(self.exog_cols)

        sigmas2 = estimate_ar_residual_variances(clean, self.endog_cols, self.max_lag)
        k = X.shape[1]
        B_post = np.zeros((k, n))

        for i in range(n):
            y_i = Y[:, i]

            b0, V0 = minnesota_prior_one_equation(
                n=n,
                p=self.max_lag,
                n_exog=n_exog,
                eq_i=i,
                sigmas2=sigmas2,
                lam1=self.lam1,
                lam2=self.lam2,
                lam3=self.lam3,
                exog_var=self.exog_var,
                prior_mean_own_lag1=0.0,
            )

            V0_inv = inv(V0)
            V_post = inv(V0_inv + X.T @ X / sigmas2[i])
            b_post = V_post @ (V0_inv @ b0 + (X.T @ y_i) / sigmas2[i])
            B_post[:, i] = b_post

        self.B_post = B_post
        self.sigmas2_ = sigmas2
        return self

    def forecast(self, h: int, future_exog: pd.DataFrame | None = None) -> pd.DataFrame:
        if self.history_ is None or self.B_post is None:
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
            y_hat = x @ self.B_post
            forecasts.append(y_hat)
            endog_values = np.vstack([endog_values, y_hat])

        return pd.DataFrame(forecasts, columns=self.endog_cols)