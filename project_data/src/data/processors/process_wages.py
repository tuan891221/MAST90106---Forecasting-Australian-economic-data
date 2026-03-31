import pandas as pd

from src.utils.paths import RAW_DIR, CURATED_DIR
from src.utils.logger import get_logger

logger = get_logger("data.processors.wages")

TARGET_COL = 6


def process_wages():
    file_path = RAW_DIR / "wages_raw.xlsx"

    if not file_path.exists():
        logger.info("Wages raw file missing. Skipping processor.")
        return

    df = pd.read_excel(file_path, sheet_name="Data1", header=None)

    data = df.iloc[10:, [0, TARGET_COL]].copy()
    data.columns = ["date", "wages_level"]

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["wages_level"] = pd.to_numeric(data["wages_level"], errors="coerce")

    data = data.dropna().sort_values("date").reset_index(drop=True)

    data["wages"] = data["wages_level"].pct_change() * 100
    data = data.dropna().reset_index(drop=True)

    if data.empty:
        raise ValueError("Wages processor produced an empty dataset")

    output_path = CURATED_DIR / "wages.csv"
    data[["date", "wages"]].to_csv(output_path, index=False)

    logger.info("Saved curated wages to %s", output_path)


def main():
    process_wages()


if __name__ == "__main__":
    main()