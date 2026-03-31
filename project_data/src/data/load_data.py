from __future__ import annotations

import pandas as pd

from src.utils.paths import CURATED_DIR
from src.utils.logger import get_logger

logger = get_logger("data.load_data")


def load_one(file_name: str, value_name: str) -> pd.DataFrame:
    file_path = CURATED_DIR / file_name
    logger.info("Loading curated data for %s from %s", value_name, file_path)

    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")

    df = df[["date", value_name]].dropna().sort_values("date").reset_index(drop=True)
    return df


def load_all_curated_data() -> dict[str, pd.DataFrame]:
    return {
        "cash_rate": load_one("cash_rate.csv", "cash_rate"),
        "inflation": load_one("inflation.csv", "inflation"),
        "output": load_one("output.csv", "output"),
        "unemployment": load_one("unemployment.csv", "unemployment"),
        "wages": load_one("wages.csv", "wages"),
    }