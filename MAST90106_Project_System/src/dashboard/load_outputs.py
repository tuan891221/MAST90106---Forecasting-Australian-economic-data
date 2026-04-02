from __future__ import annotations

import pandas as pd

from src.utils.paths import (
    QUARTERLY_MODEL_INPUT_PATH,
    FORECAST_ALL_PATH,
    FUTURE_FORECAST_PATH,
    METRICS_ALL_PATH,
    OVERVIEW_SUMMARY_PATH,
    BEST_MODELS_PATH,
    MODEL_RANKING_PATH,
)


def load_quarterly_model_input() -> pd.DataFrame:
    return pd.read_csv(QUARTERLY_MODEL_INPUT_PATH, parse_dates=["date"])


def load_rolling_forecasts() -> pd.DataFrame:
    return pd.read_csv(FORECAST_ALL_PATH, parse_dates=["date"])


def load_future_forecasts() -> pd.DataFrame:
    return pd.read_csv(FUTURE_FORECAST_PATH, parse_dates=["date"])


def load_metrics() -> pd.DataFrame:
    return pd.read_csv(METRICS_ALL_PATH)


def load_overview_summary() -> pd.DataFrame:
    return pd.read_csv(OVERVIEW_SUMMARY_PATH)


def load_best_models() -> pd.DataFrame:
    return pd.read_csv(BEST_MODELS_PATH)


def load_model_ranking() -> pd.DataFrame:
    return pd.read_csv(MODEL_RANKING_PATH)


def load_model_input() -> pd.DataFrame:
    return load_quarterly_model_input()


def load_forecasts() -> pd.DataFrame:
    return load_rolling_forecasts()


def load_single_variable_history_and_forecast(
    variable: str,
    model: str,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hist = load_quarterly_model_input()
    roll = load_rolling_forecasts()
    future = load_future_forecasts()

    hist_sub = hist[["date", variable]].copy()
    hist_sub = hist_sub.rename(columns={variable: "actual"})

    roll_sub = roll[
        (roll["variable"] == variable)
        & (roll["model"] == model)
        & (roll["horizon"] == horizon)
    ][["date", "forecast"]].copy()

    future_sub = future[
        (future["variable"] == variable)
        & (future["model"] == model)
    ][["date", "horizon", "forecast"]].copy()

    return hist_sub, roll_sub, future_sub


def load_combined_history_and_forecast(
    model: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hist = load_quarterly_model_input()
    future = load_future_forecasts()

    variables = ["output", "inflation", "cash_rate", "unemployment", "wages"]

    hist_long = hist[["date"] + variables].melt(
        id_vars="date",
        value_vars=variables,
        var_name="variable",
        value_name="raw_value",
    )

    future_long = future[
        future["model"] == model
    ][["date", "variable", "forecast"]].copy()
    future_long = future_long.rename(columns={"forecast": "raw_value"})

    stats = hist_long.groupby("variable")["raw_value"].agg(["mean", "std"]).reset_index()

    hist_long = hist_long.merge(stats, on="variable", how="left")
    hist_long["value"] = hist_long.apply(
        lambda r: 0.0 if pd.isna(r["std"]) or r["std"] == 0 else (r["raw_value"] - r["mean"]) / r["std"],
        axis=1,
    )

    future_long = future_long.merge(stats, on="variable", how="left")
    future_long["value"] = future_long.apply(
        lambda r: 0.0 if pd.isna(r["std"]) or r["std"] == 0 else (r["raw_value"] - r["mean"]) / r["std"],
        axis=1,
    )

    return hist_long[["date", "variable", "value"]], future_long[["date", "variable", "value"]]