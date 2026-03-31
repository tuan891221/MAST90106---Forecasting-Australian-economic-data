from __future__ import annotations

import pandas as pd

from src.data.sources.download_cash_rate_raw import main as download_cash_rate
from src.data.sources.download_inflation_raw import main as download_inflation
from src.data.sources.download_output_raw import main as download_output
from src.data.sources.download_unemployment_raw import main as download_unemployment
from src.data.sources.download_wages_raw import main as download_wages

from src.data.processors.process_cash_rate import main as process_cash_rate
from src.data.processors.process_inflation import main as process_inflation
from src.data.processors.process_output import main as process_output
from src.data.processors.process_unemployment import main as process_unemployment
from src.data.processors.process_wages import main as process_wages

from src.data.load_data import load_all_curated_data
from src.data.resample_to_quarterly import resample_all_to_quarterly
from src.data.merge_data import merge_quarterly_datasets

from src.utils.paths import MERGED_DIR
from src.utils.config_loader import load_config


OUTPUT_FILE = MERGED_DIR / "all_variables.csv"


def enforce_continuous_quarter_index(df: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config()
    start = pd.to_datetime(cfg["data"]["start_date"]) + pd.offsets.QuarterEnd(0)
    end = pd.to_datetime(cfg["data"]["end_date"]) + pd.offsets.QuarterEnd(0)

    full_quarters = pd.DataFrame({
        "date": pd.date_range(start=start, end=end, freq="QE")
    })

    out = full_quarters.merge(df, on="date", how="left")
    out = out.sort_values("date").reset_index(drop=True)
    return out


def main() -> None:
    print("Step 1: Download raw data...")
    download_cash_rate()
    download_inflation()
    download_output()
    download_unemployment()
    download_wages()

    print("Step 2: Process raw data...")
    process_cash_rate()
    process_inflation()
    process_output()
    process_unemployment()
    process_wages()

    print("Step 3: Load curated datasets...")
    datasets = load_all_curated_data()

    print("Step 4: Resample to quarterly...")
    quarterly_datasets = resample_all_to_quarterly(datasets)

    print("Step 5: Merge datasets...")
    merged = merge_quarterly_datasets(quarterly_datasets)

    print("Step 6: Enforce continuous quarterly dates...")
    merged = enforce_continuous_quarter_index(merged)

    print("Step 7: Save merged csv...")
    merged.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved merged dataset to: {OUTPUT_FILE}")
    print("\nShape:", merged.shape)
    print("\nMissing values by column:")
    print(merged.isna().sum())


if __name__ == "__main__":
    main()