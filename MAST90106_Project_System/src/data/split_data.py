from __future__ import annotations

import pandas as pd

from src.utils.config_loader import load_config


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config()
    ratio = float(cfg["forecast"]["train_ratio"])
    split_idx = int(len(df) * ratio)
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    return train_df, test_df