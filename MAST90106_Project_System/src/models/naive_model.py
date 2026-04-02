from __future__ import annotations

import pandas as pd

from src.models.base_model import BaseTimeSeriesModel
from src.utils.constants import VARIABLES


class NaiveModel(BaseTimeSeriesModel):
    """
    Multivariate-system compatible naive model:
    repeat the last observed endogenous row for h steps.
    """

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config=config, max_lag=1)

    def fit(self, data: pd.DataFrame) -> "NaiveModel":
        self.history_ = data.copy()
        self.endog_cols = [c for c in VARIABLES if c in data.columns]
        self.exog_cols = [c for c in data.columns if c not in self.endog_cols]
        return self

    def forecast(self, h: int, future_exog: pd.DataFrame | None = None) -> pd.DataFrame:
        if self.history_ is None:
            raise ValueError("Model must be fitted before forecasting.")

        last_endog = self.history_[self.endog_cols].iloc[[-1]].copy()
        out = pd.concat([last_endog] * h, ignore_index=True)
        out.columns = self.endog_cols
        return out