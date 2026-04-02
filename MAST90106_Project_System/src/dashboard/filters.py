from __future__ import annotations

import pandas as pd


def filter_forecasts(df: pd.DataFrame, variable: str, horizon: int, models: list[str]) -> pd.DataFrame:
    out = df[(df["variable"] == variable) & (df["horizon"] == horizon)].copy()
    if models:
        out = out[out["model"].isin(models)]
    return out


def filter_metrics(df: pd.DataFrame, variable: str | None = None, horizon: int | None = None):
    out = df.copy()
    if variable is not None:
        out = out[out["variable"] == variable]
    if horizon is not None:
        out = out[out["horizon"] == horizon]
    return out