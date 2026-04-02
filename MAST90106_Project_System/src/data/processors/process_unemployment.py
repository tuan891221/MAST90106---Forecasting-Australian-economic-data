import pandas as pd
from src.utils.paths import RAW_ABS_DIR, CURATED_DIR


TARGET_COL = 64  # Unemployment rate ; Persons ; Trend ; series A84423134K


def process_unemployment():
    file_path = RAW_ABS_DIR / "unemployment_raw.xlsx"

    df = pd.read_excel(file_path, sheet_name="Data1", header=None)

    # row 10 onward is data
    data = df.iloc[10:, [0, TARGET_COL]].copy()
    data.columns = ["date", "unemployment_rate"]

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["unemployment_rate"] = pd.to_numeric(data["unemployment_rate"], errors="coerce")

    data = data.dropna().sort_values("date").reset_index(drop=True)

    if data.empty:
        raise ValueError("Unemployment processor produced an empty dataset")

    output_path = CURATED_DIR / "unemployment" / "unemployment.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)

    print("Unemployment processed:", output_path)
    print(data.head())


def main():
    process_unemployment()