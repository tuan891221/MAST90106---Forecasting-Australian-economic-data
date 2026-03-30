from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecasting.horizon_manager import get_target_column
from src.forecasting.rolling_forecast import build_feature_matrix
from src.models.model_registry import get_model
from src.utils.constants import VARIABLES
from src.utils.logger import get_logger

logger = get_logger("forecast.pipeline")


def run_forecast_pipeline(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    enabled_models = config["models"]["enabled"]
    horizons = config["forecast"]["horizons"]
    min_train = int(config["forecast"]["min_train_periods"])

    all_variables = [v for v in VARIABLES if v in df.columns]
    results = []

    for variable in all_variables:
        for horizon in horizons:
            target_col = get_target_column(variable, horizon)

            for model_name in enabled_models:
                model = get_model(model_name, config)
                max_lag = model.max_lag

                feat_df, feature_names = build_feature_matrix(
                    df=df,
                    target_variable=variable,
                    all_variables=all_variables,
                    max_lag=max_lag,
                    model_name=model_name,
                    include_covid_dummy=(config["covid"]["enabled"] and config["covid"]["robustness"]["use_dummy"]),
                )

                work = pd.concat([df[["date", variable, target_col]], feat_df.drop(columns=["date"])], axis=1)
                work = work.dropna().reset_index(drop=True)

                if len(work) <= min_train:
                    logger.info("Skipping %s/%s/%s due to insufficient data", variable, horizon, model_name)
                    continue

                feature_cols = [c for c in work.columns if c not in {"date", variable, target_col}]
                for i in range(min_train, len(work)):
                    train = work.iloc[:i].copy()
                    test = work.iloc[[i]].copy()

                    X_train = train[feature_cols].to_numpy(dtype=float)
                    y_train = train[target_col].to_numpy(dtype=float)
                    X_test = test[feature_cols].to_numpy(dtype=float)

                    model.fit(X_train, y_train, feature_names=feature_cols)
                    y_pred = float(model.predict(X_test)[0])
                    y_true = float(test[target_col].iloc[0])

                    results.append(
                        {
                            "date": test["date"].iloc[0],
                            "variable": variable,
                            "horizon": horizon,
                            "model": model_name,
                            "actual": y_true,
                            "forecast": y_pred,
                            "error": y_pred - y_true,
                            "is_covid_period": int(test["is_covid_period"].iloc[0]) if "is_covid_period" in test.columns else 0,
                        }
                    )

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df["date"] = pd.to_datetime(result_df["date"])
    return result_df