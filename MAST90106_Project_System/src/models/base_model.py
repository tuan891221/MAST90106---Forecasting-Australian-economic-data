from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseTimeSeriesModel(ABC):
    def __init__(self, config: dict | None = None, max_lag: int = 4) -> None:
        self.config = config or {}
        self.max_lag = int(max_lag)

        self.endog_cols: list[str] = []
        self.exog_cols: list[str] = []
        self.history_: pd.DataFrame | None = None

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> "BaseTimeSeriesModel":
        raise NotImplementedError

    @abstractmethod
    def forecast(self, h: int, future_exog: pd.DataFrame | None = None) -> pd.DataFrame:
        raise NotImplementedError