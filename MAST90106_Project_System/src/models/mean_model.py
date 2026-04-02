from __future__ import annotations

import pandas as pd

from src.models.base_model import BaseTimeSeriesModel
from src.utils.constants import VARIABLES


class MeanModel(BaseTimeSeriesModel):
    """
    Multivariate-system compatible mean model:
    repeat historical mean of each endogenous variable.
    """

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config=config, max_lag=1)
        self.mean_: pd.Series | None = None

    def fit(self, data: pd.DataFrame) -> "MeanModel":
        self.history_ = data.copy()
        self.endog_cols = [c for c in VARIABLES if c in data.columns]
        self.exog_cols = [c for c in data.columns if c not in self.endog_cols]
        self.mean_ = data[self.endog_cols].mean()
        return self

    def forecast(self, h: int, future_exog: pd.DataFrame | None = None) -> pd.DataFrame:
        if self.mean_ is None:
            raise ValueError("Model must be fitted before forecasting.")

        out = pd.DataFrame([self.mean_.values] * h, columns=self.endog_cols)
        return out