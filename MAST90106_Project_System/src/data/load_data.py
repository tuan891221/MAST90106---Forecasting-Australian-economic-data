from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.paths import (
    CURATED_CASH_RATE_DIR,
    CURATED_INFLATION_DIR,
    CURATED_OUTPUT_DIR,
    CURATED_UNEMPLOYMENT_DIR,
    CURATED_WAGES_DIR,
)

logger = get_logger("data.load_data")

CURATED_FILE_CANDIDATES = {
    "inflation": [
        CURATED_INFLATION_DIR / "trimmed_mean_inflation_2000_2025.csv",
        CURATED_INFLATION_DIR / "inflation_quarterly.csv",
    ],
    "cash_rate": [
        CURATED_CASH_RATE_DIR / "cash_rate.csv",
        CURATED_CASH_RATE_DIR / "cash_rate_quarterly.csv",
    ],
    "output": [
        CURATED_OUTPUT_DIR / "output_growth_quarterly.csv",
        CURATED_OUTPUT_DIR / "output.csv",
    ],
    "unemployment": [
        CURATED_UNEMPLOYMENT_DIR / "unemployment.csv",
        CURATED_UNEMPLOYMENT_DIR / "unemployment_quarterly.csv",
    ],
    "wages": [
        CURATED_WAGES_DIR / "wages_growth_quarterly.csv",
        CURATED_WAGES_DIR / "wages.csv",
    ],
}

STANDARD_VALUE_COLUMN_MAP = {
    "inflation": ["trimmed_mean_inflation_qoq", "inflation", "value"],
    "cash_rate": ["cash_rate", "value"],
    "output": ["output_growth", "value"],
    "unemployment": ["unemployment_rate", "value"],
    "wages": ["wages_growth", "value"],
}


def _find_existing_file(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def _find_value_column(df: pd.DataFrame, variable: str) -> str:
    lowered = {str(col).strip().lower(): col for col in df.columns}
    for candidate in STANDARD_VALUE_COLUMN_MAP[variable]:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    non_date = [c for c in df.columns if str(c).strip().lower() != "date"]
    if len(non_date) == 1:
        return non_date[0]
    raise ValueError(f"Could not identify value column for {variable} from {list(df.columns)}")


def _standardize(path: Path, variable: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    lowered = {str(col).strip().lower(): col for col in df.columns}
    if "date" not in lowered:
        raise ValueError(f"'date' column missing in {path}")
    date_col = lowered["date"]
    value_col = _find_value_column(df, variable)

    out = df[[date_col, value_col]].copy()
    out.columns = ["date", variable]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out[variable] = pd.to_numeric(out[variable], errors="coerce")
    out = out.dropna().sort_values("date").reset_index(drop=True)
    return out


def load_all_curated_data() -> dict[str, pd.DataFrame]:
    cfg = load_config()
    strict = bool(cfg["project"].get("strict_required_variables", True))
    variables = list(cfg["data"]["variables"].keys())

    datasets: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for variable in variables:
        path = _find_existing_file(CURATED_FILE_CANDIDATES[variable])
        if path is None:
            missing.append(variable)
            continue
        logger.info("Loading curated data for %s from %s", variable, path)
        datasets[variable] = _standardize(path, variable)

    if missing and strict:
        raise FileNotFoundError(
            f"Missing curated datasets for: {missing}. "
            "Either provide raw files for processors or disable strict_required_variables."
        )

    if not datasets:
        raise FileNotFoundError("No curated datasets found at all.")

    return datasets