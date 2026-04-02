from __future__ import annotations

import pandas as pd


def rank_models(metrics_df: pd.DataFrame) -> pd.DataFrame:
    out = metrics_df.sort_values(["variable", "horizon", "rmse"]).copy()
    out["rank"] = out.groupby(["variable", "horizon"])["rmse"].rank(method="dense")
    return out.reset_index(drop=True)