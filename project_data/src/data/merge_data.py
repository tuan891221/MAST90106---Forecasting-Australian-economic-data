from __future__ import annotations

from functools import reduce
import pandas as pd

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("data.merge_data")


def merge_quarterly_datasets(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = [df[["date", name]].copy() for name, df in datasets.items()]
    merged = reduce(lambda l, r: pd.merge(l, r, on="date", how="outer"), frames)
    merged = merged.sort_values("date").reset_index(drop=True)

    cfg = load_config()
    start = pd.to_datetime(cfg["data"]["start_date"]) + pd.offsets.QuarterEnd(0)
    end = pd.to_datetime(cfg["data"]["end_date"]) + pd.offsets.QuarterEnd(0)

    merged = merged[(merged["date"] >= start) & (merged["date"] <= end)].copy()

    logger.info("Merged shape: %s", merged.shape)
    return merged