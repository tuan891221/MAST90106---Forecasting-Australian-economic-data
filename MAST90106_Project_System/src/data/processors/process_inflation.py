from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger
from src.utils.paths import CURATED_INFLATION_DIR, RAW_RBA_DIR

logger = get_logger("data.processors.inflation")

RAW_FILE = RAW_RBA_DIR / "g01hist.xlsx"
OUTPUT_FILE = CURATED_INFLATION_DIR / "trimmed_mean_inflation_2000_2025.csv"


def find_trimmed_mean_column(df: pd.DataFrame) -> str:
    candidates = []
    for col in df.columns:
        col_str = str(col).strip().lower()
        if "trimmed" in col_str and "quarter" in col_str:
            candidates.append(col)
    if not candidates:
        raise ValueError("Could not detect trimmed mean quarterly column.")
    return candidates[0]


def load_raw_excel(file_path: Path) -> pd.DataFrame:
    raw = pd.read_excel(file_path, header=None)
    header_row = None
    for i in range(min(30, len(raw))):
        row_text = " | ".join(raw.iloc[i].fillna("").astype(str).tolist()).lower()
        if "trimmed" in row_text and "weighted median" in row_text:
            header_row = i
            break
    if header_row is None:
        raise ValueError("Could not detect header row in inflation raw file.")
    return pd.read_excel(file_path, header=header_row)


def clean_inflation_data(df: pd.DataFrame) -> pd.DataFrame:
    date_col = df.columns[0]
    value_col = find_trimmed_mean_column(df)
    out = df[[date_col, value_col]].copy()
    out.columns = ["date", "trimmed_mean_inflation_qoq"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["trimmed_mean_inflation_qoq"] = pd.to_numeric(
        out["trimmed_mean_inflation_qoq"], errors="coerce"
    )
    out = out.dropna()
    out = out[(out["date"] >= "2000-01-01") & (out["date"] <= "2025-12-31")]
    out = out.sort_values("date").reset_index(drop=True)
    return out


def main() -> None:
    if not RAW_FILE.exists():
        logger.info("Inflation raw file missing. Skipping processor.")
        return
    CURATED_INFLATION_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw_excel(RAW_FILE)
    out = clean_inflation_data(df)
    out.to_csv(OUTPUT_FILE, index=False)
    logger.info("Saved curated inflation to %s", OUTPUT_FILE)