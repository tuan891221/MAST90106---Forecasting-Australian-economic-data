from __future__ import annotations

import pandas as pd

from src.utils.paths import FORECAST_ALL_PATH, METRICS_ALL_PATH, MODEL_RANKING_PATH, QUARTERLY_MODEL_INPUT_PATH


def load_forecasts() -> pd.DataFrame:
    return pd.read_csv(FORECAST_ALL_PATH, parse_dates=["date"])


def load_metrics() -> pd.DataFrame:
    return pd.read_csv(METRICS_ALL_PATH)


def load_rankings() -> pd.DataFrame:
    return pd.read_csv(MODEL_RANKING_PATH)


def load_model_input() -> pd.DataFrame:
    return pd.read_csv(QUARTERLY_MODEL_INPUT_PATH, parse_dates=["date"])