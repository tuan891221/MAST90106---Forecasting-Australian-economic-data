from __future__ import annotations

import pandas as pd
from src.utils.paths import RAW_ABS_DIR, CURATED_DIR


# Candidate column for:
# Non-farm GDP, chain volume measures, percentage changes, seasonally adjusted
TARGET_COL = 118


def process_output():
    """
    Process real (chain volume) non-farm output growth from ABS.
    """

    file_path = RAW_ABS_DIR / "output_raw.xlsx"
    df = pd.read_excel(file_path, sheet_name="Data1", header=None)

    # Print metadata for verification
    print("Description:", df.iloc[0, TARGET_COL])
    print("Unit:", df.iloc[1, TARGET_COL])
    print("Series type:", df.iloc[2, TARGET_COL])
    print("Series ID:", df.iloc[9, TARGET_COL])

    # Extract observations
    data = df.iloc[10:, [0, TARGET_COL]].copy()
    data.columns = ["date", "output"]

    # Convert types
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["output"] = pd.to_numeric(data["output"], errors="coerce")

    # Keep valid observations only
    data = data.dropna().sort_values("date").reset_index(drop=True)

    if data.empty:
        raise ValueError("Output processor produced an empty dataset")

    output_path = CURATED_DIR / "output" / "output_growth_quarterly.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)

    print("Output processed:", output_path)
    print(data.tail(12))


def main():
    process_output()