from __future__ import annotations

import numpy as np
import pandas as pd


def build_feature_matrix(
    df: pd.DataFrame,
    target_variable: str,
    all_variables: list[str],
    max_lag: int,
    model_name: str,
    include_covid_dummy: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    out = df[["date"]].copy()
    feature_names: list[str] = []

    if model_name in {"naive", "mean", "ar"}:
        feature_vars = [target_variable]
    else:
        feature_vars = all_variables

    for var in feature_vars:
        for lag in range(1, max_lag + 1):
            col = f"{var}_lag{lag}"
            out[col] = df[var].shift(lag)
            feature_names.append(col)

    if include_covid_dummy and "is_covid_period" in df.columns:
        out["is_covid_period"] = df["is_covid_period"]
        feature_names.append("is_covid_period")

    return out, feature_names