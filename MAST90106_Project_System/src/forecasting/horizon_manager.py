from __future__ import annotations


def get_target_column(variable: str, horizon: int) -> str:
    if horizon not in (1, 4):
        raise ValueError("Supported horizons are 1 and 4.")
    return f"{variable}_target_{horizon}q"