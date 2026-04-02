from __future__ import annotations

import pandas as pd

from src.utils.config_loader import load_config


def add_covid_flags(df: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config()
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    if not cfg["covid"]["enabled"]:
        out["is_covid_period"] = 0
        out["is_post_covid_period"] = 0
        return out

    start = pd.to_datetime(cfg["covid"]["start"])
    end = pd.to_datetime(cfg["covid"]["end"])
    out["is_covid_period"] = ((out["date"] >= start) & (out["date"] <= end)).astype(int)
    out["is_post_covid_period"] = (out["date"] > end).astype(int)
    return out