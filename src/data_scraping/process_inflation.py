import pandas as pd
from pathlib import Path

from src.utils.paths import RAW_DIR, CURATED_DIR


RAW_FILE = RAW_DIR / "g01hist.xlsx"
OUTPUT_FILE = CURATED_DIR / "trimmed_mean_inflation_2000_2025.csv"


def find_trimmed_mean_column(df: pd.DataFrame) -> str:
    """
    Automatically find the column corresponding to quarterly trimmed mean inflation
    """
    candidates = []

    for col in df.columns:
        col_str = str(col).strip().lower()

        if "trimmed" in col_str and "quarter" in col_str:
            candidates.append(col)

    if not candidates:
        raise ValueError(
            "No column containing 'trimmed' and 'quarter' was found. Please check the structure of the raw Excel file."
        )

    return candidates[0]


def load_raw_excel(file_path: Path) -> pd.DataFrame:
    """
    Read the RBA G1 Excel file and automatically skip metadata rows at the top
    """
    raw = pd.read_excel(file_path, header=None)

    header_row = None
    for i in range(min(30, len(raw))):
        row_values = raw.iloc[i].fillna("").astype(str).tolist()
        joined = " | ".join(row_values).lower()

        if "trimmed" in joined and "weighted median" in joined:
            header_row = i
            break

    if header_row is None:
        raise ValueError(
            "Header row could not be detected. Please inspect the structure of g01hist.xlsx manually."
        )

    df = pd.read_excel(file_path, header=header_row)
    return df


def clean_inflation_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the inflation dataset and retain only the quarterly trimmed mean inflation
    """
    # Assume the first column is the date
    date_col = df.columns[0]
    trimmed_col = find_trimmed_mean_column(df)

    cleaned = df[[date_col, trimmed_col]].copy()
    cleaned.columns = ["date", "trimmed_mean_inflation_qoq"]

    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned["trimmed_mean_inflation_qoq"] = pd.to_numeric(
        cleaned["trimmed_mean_inflation_qoq"], errors="coerce"
    )

    cleaned = cleaned.dropna(subset=["date", "trimmed_mean_inflation_qoq"])

    cleaned = cleaned[
        (cleaned["date"] >= "2000-01-01") &
        (cleaned["date"] <= "2025-12-31")
    ].copy()

    cleaned = cleaned.sort_values("date").reset_index(drop=True)

    return cleaned


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw file not found: {RAW_FILE}")

    CURATED_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_excel(RAW_FILE)
    cleaned = clean_inflation_data(df)

    cleaned.to_csv(OUTPUT_FILE, index=False)

    print("Processing completed!")
    print(f"Raw file: {RAW_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(cleaned.head())
    print(cleaned.tail())
    print(f"Number of rows: {len(cleaned)}")


if __name__ == "__main__":
    main()