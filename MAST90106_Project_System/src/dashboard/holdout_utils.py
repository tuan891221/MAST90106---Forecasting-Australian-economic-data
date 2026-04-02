from __future__ import annotations

import pandas as pd

from src.models.model_registry import get_model
from src.utils.constants import VARIABLES


def _get_model_input_columns(df: pd.DataFrame, include_covid: bool) -> tuple[list[str], list[str]]:
    endog_cols = [v for v in VARIABLES if v in df.columns]
    exog_cols: list[str] = []

    if include_covid:
        if "is_covid_period" in df.columns:
            exog_cols.append("is_covid_period")
        if "is_post_covid_period" in df.columns:
            exog_cols.append("is_post_covid_period")

    return endog_cols, exog_cols


def build_holdout_forecast_view(
    df: pd.DataFrame,
    config: dict,
    model_name: str,
    test_size: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

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

    if len(work) <= test_size + 12:
        raise ValueError("Not enough observations for holdout forecast view.")

    train = work.iloc[:-test_size].copy()
    test = work.iloc[-test_size:].copy()

    # Full model for holdout test forecast
    model = get_model(model_name, config)

    if model_name in {"ar", "var", "bvar", "factor"}:
        model.fit(train[endog_cols + exog_cols].copy())
    else:
        model.fit(train[endog_cols].copy())

    # In-sample expanding fitted values
    fitted_rows = []

    base_lag = int(getattr(model, "max_lag", 4))

    # BVAR / VAR / Factor need a more conservative minimum sample size
    if model_name == "bvar":
        min_start = max(20, base_lag * 4)
    elif model_name in {"var", "factor"}:
        min_start = max(16, base_lag * 3)
    elif model_name == "ar":
        min_start = max(10, base_lag * 2)
    else:
        min_start = 5

    for train_end in range(min_start, len(train)):
        sub_train = train.iloc[:train_end].copy()
        actual_date = pd.to_datetime(train.iloc[train_end]["date"])

        sub_model = get_model(model_name, config)

        try:
            if model_name in {"ar", "var", "bvar", "factor"}:
                sub_model.fit(sub_train[endog_cols + exog_cols].copy())
                future_exog = (
                    train.iloc[train_end : train_end + 1][exog_cols].copy().reset_index(drop=True)
                    if exog_cols
                    else None
                )
                pred_df = sub_model.forecast(1, future_exog=future_exog)
            else:
                sub_model.fit(sub_train[endog_cols].copy())
                pred_df = sub_model.forecast(1)
        except Exception:
            continue

        pred_row = pred_df.iloc[0]
        row = {"date": actual_date}
        for v in endog_cols:
            row[v] = float(pred_row[v])
        fitted_rows.append(row)

    fitted_df = pd.DataFrame(fitted_rows)

    # Held-out test forecast
    if model_name in {"ar", "var", "bvar", "factor"}:
        future_exog = test[exog_cols].copy().reset_index(drop=True) if exog_cols else None
        fcst_df = model.forecast(test_size, future_exog=future_exog)
    else:
        fcst_df = model.forecast(test_size)

    forecast_df = pd.DataFrame({"date": pd.to_datetime(test["date"]).values})
    for v in endog_cols:
        forecast_df[v] = fcst_df[v].values

    actual_df = work[["date"] + endog_cols].copy()

    return actual_df, fitted_df, forecast_df