from __future__ import annotations

import pandas as pd

from src.models.model_registry import get_model
from src.utils.constants import VARIABLES
from src.utils.logger import get_logger

logger = get_logger("forecast.future")


def _next_quarter_end(last_date: pd.Timestamp, steps: int = 1) -> pd.Timestamp:
    return pd.Timestamp(last_date) + pd.offsets.QuarterEnd(steps)


def _get_model_input_columns(df: pd.DataFrame, include_covid: bool) -> tuple[list[str], list[str]]:
    endog_cols = [v for v in VARIABLES if v in df.columns]
    exog_cols: list[str] = []

    if include_covid:
        if "is_covid_period" in df.columns:
            exog_cols.append("is_covid_period")
        if "is_post_covid_period" in df.columns:
            exog_cols.append("is_post_covid_period")

    return endog_cols, exog_cols


def _build_future_exog_path(last_row: pd.Series, exog_cols: list[str], h: int) -> pd.DataFrame:
    if not exog_cols:
        return pd.DataFrame(index=range(h))

    rows = []
    for _ in range(h):
        row = {}
        for col in exog_cols:
            if col == "is_covid_period":
                row[col] = 0
            elif col == "is_post_covid_period":
                row[col] = 1
            else:
                row[col] = last_row[col]
        rows.append(row)

    return pd.DataFrame(rows, columns=exog_cols)


def run_future_forecast(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    enabled_models = config["models"]["enabled"]
    steps_ahead = int(config.get("forecast", {}).get("future_steps", 8))

    use_covid_dummy = (
        config.get("covid", {}).get("enabled", False)
        and config.get("covid", {}).get("robustness", {}).get("use_dummy", False)
    )

    endog_cols, exog_cols = _get_model_input_columns(df, include_covid=use_covid_dummy)
    use_cols = ["date"] + endog_cols + exog_cols

    work = df[use_cols].copy()
    for col in endog_cols + exog_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=endog_cols).reset_index(drop=True)

    last_hist_date = pd.to_datetime(work["date"]).max()
    last_row = work.iloc[-1]

    results = []

    for model_name in enabled_models:
        model = get_model(model_name, config)

        try:
            if model_name in {"var", "bvar", "factor", "ar"}:
                train_df = work[endog_cols + exog_cols].copy()
                future_exog = _build_future_exog_path(last_row, exog_cols, steps_ahead)
                model.fit(train_df)
                forecast_path = model.forecast(steps_ahead, future_exog=future_exog)
            else:
                train_df = work[endog_cols].copy()
                model.fit(train_df)
                forecast_path = model.forecast(steps_ahead)
        except Exception as e:
            logger.warning("Future forecast failed for model=%s: %s", model_name, e)
            continue

        for step in range(1, steps_ahead + 1):
            forecast_date = _next_quarter_end(last_hist_date, steps=step)
            row = forecast_path.iloc[step - 1]

            for variable in endog_cols:
                results.append(
                    {
                        "date": forecast_date,
                        "variable": variable,
                        "model": model_name,
                        "horizon": step,
                        "forecast": float(row[variable]),
                    }
                )

    out = pd.DataFrame(results)
    if not out.empty:
        out["date"] = pd.to_datetime(out["date"])
    return out