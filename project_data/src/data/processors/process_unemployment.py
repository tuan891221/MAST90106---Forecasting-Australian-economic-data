import pandas as pd

from src.utils.paths import RAW_DIR, CURATED_DIR
from src.utils.logger import get_logger

logger = get_logger("data.processors.unemployment")

TARGET_COL = 64


def process_unemployment():
    file_path = RAW_DIR / "unemployment_raw.xlsx"

    if not file_path.exists():
        logger.info("Unemployment raw file missing. Skipping processor.")
        return

    df = pd.read_excel(file_path, sheet_name="Data1", header=None)

    data = df.iloc[10:, [0, TARGET_COL]].copy()
    data.columns = ["date", "unemployment"]

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["unemployment"] = pd.to_numeric(data["unemployment"], errors="coerce")

    data = data.dropna().sort_values("date").reset_index(drop=True)

    if data.empty:
        raise ValueError("Unemployment processor produced an empty dataset")

    output_path = CURATED_DIR / "unemployment.csv"
    data.to_csv(output_path, index=False)

    logger.info("Saved curated unemployment to %s", output_path)


def main():
    process_unemployment()


if __name__ == "__main__":
    main()