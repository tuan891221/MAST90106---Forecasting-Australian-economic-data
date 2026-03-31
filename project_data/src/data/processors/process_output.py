from __future__ import annotations

import pandas as pd

from src.utils.paths import RAW_DIR, CURATED_DIR
from src.utils.logger import get_logger

logger = get_logger("data.processors.output")

RAW_FILE = RAW_DIR / "output_raw.xlsx"
OUTPUT_FILE = CURATED_DIR / "output.csv"

TARGET_COL = 118   # DO column


def process_output() -> None:
    if not RAW_FILE.exists():
        logger.info("Output raw file missing. Skipping processor.")
        return

    df = pd.read_excel(RAW_FILE, sheet_name="Data1", header=None)

    logger.info("Output description: %s", df.iloc[0, TARGET_COL])
    logger.info("Output unit: %s", df.iloc[1, TARGET_COL])
    logger.info("Output series type: %s", df.iloc[2, TARGET_COL])
    logger.info("Output series ID: %s", df.iloc[9, TARGET_COL])

    data = df.iloc[10:, [0, TARGET_COL]].copy()
    data.columns = ["date", "output"]

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["output"] = pd.to_numeric(data["output"], errors="coerce")

    data = data.dropna().sort_values("date").reset_index(drop=True)

    data = data[
        (data["date"] >= "2000-01-01") &
        (data["date"] <= "2025-12-31")
    ].copy()

    if data.empty:
        raise ValueError("Output processor produced an empty dataset")

    data[["date", "output"]].to_csv(OUTPUT_FILE, index=False)
    logger.info("Saved curated output to %s", OUTPUT_FILE)


def main():
    process_output()


if __name__ == "__main__":
    main()