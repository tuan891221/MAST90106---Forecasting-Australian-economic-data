from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("data.targets")


def _compound_4q_growth(series: pd.Series) -> pd.Series:
    a = series.shift(-1) / 100.0
    b = series.shift(-2) / 100.0
    c = series.shift(-3) / 100.0
    d = series.shift(-4) / 100.0
    return ((1 + a) * (1 + b) * (1 + c) * (1 + d) - 1) * 100


def build_forecast_targets(df: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config()
    out = df.copy().sort_values("date").reset_index(drop=True)

    for variable, meta in cfg["data"]["variables"].items():
        if variable not in out.columns:
            continue

        target_type = meta["target_type"]
        out[f"{variable}_target_1q"] = out[variable].shift(-1)

        if target_type == "growth":
            out[f"{variable}_target_4q"] = _compound_4q_growth(out[variable])
        elif target_type == "level":
            out[f"{variable}_target_4q"] = out[variable].shift(-4)
        else:
            raise ValueError(f"Unsupported target_type {target_type}")

        out[f"{variable}_target_type"] = target_type

    return out