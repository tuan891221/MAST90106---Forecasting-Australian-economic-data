import pandas as pd
from src.utils.paths import RAW_ABS_DIR, CURATED_DIR


TARGET_COL = 6  # Private and Public ; All industries ; Seasonally Adjusted ; series A2713849C


def process_wages():
    file_path = RAW_ABS_DIR / "wages_raw.xlsx"

    df = pd.read_excel(file_path, sheet_name="Data1", header=None)

    # row 10 onward is data
    data = df.iloc[10:, [0, TARGET_COL]].copy()
    data.columns = ["date", "wages_level"]

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["wages_level"] = pd.to_numeric(data["wages_level"], errors="coerce")

    data = data.dropna().sort_values("date").reset_index(drop=True)

    data["wages_growth"] = data["wages_level"].pct_change() * 100
    data = data.dropna().reset_index(drop=True)

    if data.empty:
        raise ValueError("Wages processor produced an empty dataset")

    output_path = CURATED_DIR / "wages" / "wages_growth_quarterly.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data[["date", "wages_growth"]].to_csv(output_path, index=False)

    print("Wages processed:", output_path)
    print(data.head())


def main():
    process_wages()