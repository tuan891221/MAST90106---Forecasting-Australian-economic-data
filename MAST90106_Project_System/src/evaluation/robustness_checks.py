from __future__ import annotations

import pandas as pd

from src.evaluation.summarize_metrics import summarize_metrics


def build_robustness_outputs(forecast_df: pd.DataFrame):
    full_df = summarize_metrics(forecast_df)
    excluded_df = summarize_metrics(forecast_df[forecast_df["is_covid_period"] == 0].copy())
    dummy_df = full_df.copy()
    dummy_df["note"] = "main_pipeline_includes_covid_dummy_in_features_when_enabled"
    return full_df, excluded_df, dummy_df