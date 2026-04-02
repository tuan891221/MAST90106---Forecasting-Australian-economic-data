from __future__ import annotations

from src.utils.constants import DISPLAY_NAMES


def pretty_name(key: str) -> str:
    return DISPLAY_NAMES.get(key, key)