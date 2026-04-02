from __future__ import annotations

import pandas as pd

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("data.resample")


def _quarterly_average(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    out = df.set_index("date").sort_index().resample("Q")[value_col].mean().reset_index()
    return out


def _quarterly_last(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    out = df.set_index("date").sort_index().resample("Q")[value_col].last().reset_index()
    return out


def _as_is_quarterly(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    out = df.copy().sort_values("date").reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"]) + pd.offsets.QuarterEnd(0)
    return out[["date", value_col]]


def resample_single_dataset_to_quarterly(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    cfg = load_config()
    method = cfg["data"]["variables"][variable]["quarterly_method"]

    if method == "mean":
        out = _quarterly_average(df, variable)
    elif method == "last":
        out = _quarterly_last(df, variable)
    elif method == "as_is":
        out = _as_is_quarterly(df, variable)
    else:
        raise ValueError(f"Unsupported quarterly_method: {method}")

    return out.dropna().sort_values("date").reset_index(drop=True)


def resample_all_to_quarterly(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {
        name: resample_single_dataset_to_quarterly(df, name)
        for name, df in datasets.items()
    }