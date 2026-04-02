from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import CONFIG_PATH


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config file must be a YAML mapping.")
    return cfg