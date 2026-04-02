from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger
from src.utils.paths import CURATED_CASH_RATE_DIR, RAW_RBA_DIR

logger = get_logger("data.processors.cash_rate")
OUTPUT_FILE = CURATED_CASH_RATE_DIR / "cash_rate.csv"


def _find_raw_file() -> Path | None:
    candidates = list(RAW_RBA_DIR.glob("*cash*rate*.csv")) + list(RAW_RBA_DIR.glob("*cash*rate*.xlsx")) + list(RAW_RBA_DIR.glob("*cash*rate*.xls"))
    candidates += [RAW_RBA_DIR / "cash_rate_raw.xlsx", RAW_RBA_DIR / "cash_rate_raw.csv"]
    for p in candidates:
        if p.exists():
            return p
    return None


def _read_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def _pick_date_col(columns) -> str:
    for c in columns:
        s = str(c).lower()
        if "date" in s or "period" in s or "quarter" in s or "month" in s:
            return c
    return columns[0]


def _pick_value_col(columns) -> str:
    for c in columns:
        s = str(c).lower()
        if "cash" in s and "rate" in s:
            return c
        if "interbank" in s and "target" in s:
            return c
    return columns[1]


def main() -> None:
    raw_path = _find_raw_file()
    if raw_path is None:
        logger.info("Cash rate raw file missing. Skipping processor.")
        return

    df = _read_file(raw_path)
    date_col = _pick_date_col(df.columns)
    value_col = _pick_value_col(df.columns)

    out = df[[date_col, value_col]].copy()
    out.columns = ["date", "cash_rate"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["cash_rate"] = pd.to_numeric(out["cash_rate"], errors="coerce")
    out = out.dropna().sort_values("date").reset_index(drop=True)

    CURATED_CASH_RATE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)
    logger.info("Saved curated cash rate to %s", OUTPUT_FILE)