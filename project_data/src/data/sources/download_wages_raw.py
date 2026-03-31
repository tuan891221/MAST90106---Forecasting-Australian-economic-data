from __future__ import annotations

import requests
from requests.exceptions import RequestException

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.paths import RAW_DIR

logger = get_logger("data.sources.wages")
OUTPUT_PATH = RAW_DIR / "wages_raw.xlsx"


def main() -> None:
    cfg = load_config()
    url = cfg["data"]["raw_urls"]["wages"]

    if not url:
        logger.info("No wages URL configured. Skipping download.")
        return

    if OUTPUT_PATH.exists():
        logger.info("Wages raw already exists: %s", OUTPUT_PATH)
        return

    try:
        response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()

        with open(OUTPUT_PATH, "wb") as f:
            f.write(response.content)

        logger.info("Downloaded wages raw to %s", OUTPUT_PATH)

    except RequestException as e:
        logger.warning("Failed to download wages raw: %s", e)