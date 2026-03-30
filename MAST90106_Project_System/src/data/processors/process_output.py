import pandas as pd
from src.utils.paths import RAW_ABS_DIR, CURATED_DIR


TARGET_COL = 41  # GROSS DOMESTIC PRODUCT ; Trend ; series A2304334J


def process_output():
    file_path = RAW_ABS_DIR / "output_raw.xlsx"

    df = pd.read_excel(file_path, sheet_name="Data1", header=None)

    # row 10 onward is data
    data = df.iloc[10:, [0, TARGET_COL]].copy()
    data.columns = ["date", "gdp"]

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["gdp"] = pd.to_numeric(data["gdp"], errors="coerce")

    data = data.dropna().sort_values("date").reset_index(drop=True)

    data["output"] = data["gdp"].pct_change() * 100
    data = data.dropna().reset_index(drop=True)

    if data.empty:
        raise ValueError("Output processor produced an empty dataset")

    output_path = CURATED_DIR / "output" / "output_growth_quarterly.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data[["date", "output"]].to_csv(output_path, index=False)

    print("Output processed:", output_path)
    print(data.head())


def main():
    process_output()