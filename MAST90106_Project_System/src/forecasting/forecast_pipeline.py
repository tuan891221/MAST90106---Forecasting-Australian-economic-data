from __future__ import annotations

import pandas as pd

from src.models.model_registry import get_model
from src.utils.constants import VARIABLES
from src.utils.logger import get_logger

logger = get_logger("forecast.pipeline")


def _get_model_input_columns(df: pd.DataFrame, include_covid: bool) -> tuple[list[str], list[str]]:
    endog_cols = [v for v in VARIABLES if v in df.columns]
    exog_cols: list[str] = []

    if include_covid:
        if "is_covid_period" in df.columns:
            exog_cols.append("is_covid_period")
        if "is_post_covid_period" in df.columns:
            exog_cols.append("is_post_covid_period")

    return endog_cols, exog_cols


def run_forecast_pipeline(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    enabled_models = config["models"]["enabled"]
    horizons = config["forecast"]["horizons"]
    min_train = int(config["forecast"]["min_train_periods"])

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

    results = []

    for model_name in enabled_models:
        template_model = get_model(model_name, config)
        start_idx = max(min_train, int(getattr(template_model, "max_lag", 4)) + 4)

        for horizon in horizons:
            max_train_end = len(work) - horizon
            if max_train_end <= start_idx:
                logger.info("Skipping %s horizon=%s due to insufficient data", model_name, horizon)
                continue

            for train_end in range(start_idx, max_train_end + 1):
                train_block = work.iloc[:train_end].copy()
                actual_idx = train_end + horizon - 1
                actual_date = pd.to_datetime(work.iloc[actual_idx]["date"])
                actual_row = work.iloc[actual_idx][endog_cols]

                model = get_model(model_name, config)

                try:
                    if model_name in {"var", "bvar", "factor", "ar"}:
                        train_df = train_block[endog_cols + exog_cols].copy()
                        future_exog = (
                            work.iloc[train_end : train_end + horizon][exog_cols].copy().reset_index(drop=True)
                            if exog_cols
                            else None
                        )
                        model.fit(train_df)
                        forecast_path = model.forecast(horizon, future_exog=future_exog)
                    else:
                        train_df = train_block[endog_cols].copy()
                        model.fit(train_df)
                        forecast_path = model.forecast(horizon)
                except Exception as e:
                    logger.warning(
                        "Rolling forecast failed for model=%s horizon=%s train_end=%s: %s",
                        model_name, horizon, train_end, e
                    )
                    continue

                pred_row = forecast_path.iloc[-1]

                for variable in endog_cols:
                    actual_value = float(actual_row[variable])
                    forecast_value = float(pred_row[variable])

                    row = {
                        "date": actual_date,
                        "variable": variable,
                        "model": model_name,
                        "horizon": horizon,
                        "actual": actual_value,
                        "forecast": forecast_value,
                        "error": forecast_value - actual_value,
                    }

                    if "is_covid_period" in work.columns:
                        row["is_covid_period"] = int(work.iloc[actual_idx]["is_covid_period"])
                    if "is_post_covid_period" in work.columns:
                        row["is_post_covid_period"] = int(work.iloc[actual_idx]["is_post_covid_period"])

                    results.append(row)

    out = pd.DataFrame(results)
    if not out.empty:
        out["date"] = pd.to_datetime(out["date"])
    return out