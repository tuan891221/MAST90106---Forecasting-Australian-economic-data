from __future__ import annotations

import requests
from requests.exceptions import RequestException

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.paths import RAW_RBA_DIR

logger = get_logger("data.sources.cash_rate")
OUTPUT_PATH = RAW_RBA_DIR / "cash_rate_raw.xlsx"


def main() -> None:
    cfg = load_config()
    url = cfg["data"]["raw_urls"]["cash_rate"]

    if not url:
        logger.info("No cash_rate URL configured. Skipping download.")
        return

    if OUTPUT_PATH.exists():
        logger.info("Cash rate raw already exists: %s", OUTPUT_PATH)
        return

    RAW_RBA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(
            url,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()

        with open(OUTPUT_PATH, "wb") as f:
            f.write(response.content)

        logger.info("Downloaded cash rate raw to %s", OUTPUT_PATH)

    except RequestException as e:
        logger.warning("Failed to download cash rate raw from %s: %s", url, e)
        logger.warning("Skipping cash rate download. You can place the raw file manually in %s", OUTPUT_PATH)