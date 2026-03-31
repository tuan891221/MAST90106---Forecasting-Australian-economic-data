from __future__ import annotations

import requests
from requests.exceptions import RequestException

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.paths import RAW_DIR

logger = get_logger("data.sources.output")
OUTPUT_PATH = RAW_DIR / "output_raw.xlsx"


def main() -> None:
    cfg = load_config()
    url = cfg["data"]["raw_urls"]["output"]

    if not url:
        logger.info("No output URL configured. Skipping download.")
        return

    if OUTPUT_PATH.exists():
        logger.info("Output raw already exists: %s", OUTPUT_PATH)
        return

    try:
        response = requests.get(
            url,
            timeout=60,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()

        with open(OUTPUT_PATH, "wb") as f:
            f.write(response.content)

        logger.info("Downloaded output raw to %s", OUTPUT_PATH)

    except RequestException as e:
        logger.warning("Failed to download output raw from %s: %s", url, e)