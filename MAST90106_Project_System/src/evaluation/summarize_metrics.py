from __future__ import annotations

import pandas as pd

from src.evaluation.compute_metrics import compute_metrics


def summarize_metrics(forecast_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = forecast_df.groupby(["variable", "horizon", "model"], dropna=False)

    for (variable, horizon, model), g in grouped:
        metrics = compute_metrics(g)
        rows.append(
            {
                "variable": variable,
                "horizon": horizon,
                "model": model,
                **metrics,
                "n_obs": len(g),
            }
        )

    return pd.DataFrame(rows).sort_values(["variable", "horizon", "rmse"]).reset_index(drop=True)