from __future__ import annotations

import pandas as pd


def prepare_forecast_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out.sort_values(["date", "model"]).reset_index(drop=True)


def prepare_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["variable", "horizon", "rmse"]).reset_index(drop=True)