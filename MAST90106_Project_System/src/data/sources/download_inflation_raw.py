from __future__ import annotations

import requests

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.paths import RAW_RBA_DIR

logger = get_logger("data.sources.inflation")

OUTPUT_PATH = RAW_RBA_DIR / "g01hist.xlsx"


def download_file(url: str, output_path):
    RAW_RBA_DIR.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)


def main() -> None:
    cfg = load_config()
    url = cfg["data"]["raw_urls"]["inflation"]
    if not url:
        logger.info("No inflation URL configured. Skipping.")
        return
    if OUTPUT_PATH.exists():
        logger.info("Inflation raw already exists: %s", OUTPUT_PATH)
        return
    logger.info("Downloading inflation raw from %s", url)
    download_file(url, OUTPUT_PATH)