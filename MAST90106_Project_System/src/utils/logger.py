from __future__ import annotations

import logging

from src.utils.paths import PIPELINE_LOG_PATH


def get_logger(name: str = "macro_forecasting") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    PIPELINE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(PIPELINE_LOG_PATH, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.propagate = False
    return logger