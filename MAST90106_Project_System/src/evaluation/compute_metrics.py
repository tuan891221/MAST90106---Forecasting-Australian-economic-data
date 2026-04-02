from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    errors = df["forecast"] - df["actual"]
    return {
        "bias": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae": float(np.mean(np.abs(errors))),
    }