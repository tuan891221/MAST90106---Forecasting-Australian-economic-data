from __future__ import annotations

import inspect
import pprint

import pandas as pd

from src.data.build_targets import build_forecast_targets
from src.data.covid_flags import add_covid_flags
from src.data.load_data import load_all_curated_data
from src.data.merge_data import merge_quarterly_datasets
from src.data.processors.process_cash_rate import main as process_cash_rate_main
from src.data.processors.process_inflation import main as process_inflation_main
from src.data.processors.process_output import main as process_output_main
from src.data.processors.process_unemployment import main as process_unemployment_main
from src.data.processors.process_wages import main as process_wages_main
from src.data.resample_to_quarterly import resample_all_to_quarterly
from src.data.sources.download_cash_rate_raw import main as download_cash_rate_main
from src.data.sources.download_inflation_raw import main as download_inflation_main
from src.data.sources.download_output_raw import main as download_output_main
from src.data.sources.download_unemployment_raw import main as download_unemployment_main
from src.data.sources.download_wages_raw import main as download_wages_main
from src.data.split_data import split_train_test
from src.data.transform_variables import transform_variables

from src.evaluation.compute_metrics import compute_metrics
from src.evaluation.rank_models import rank_models
from src.evaluation.robustness_checks import build_robustness_outputs
from src.evaluation.summarize_metrics import summarize_metrics

from src.forecasting.forecast_pipeline import run_forecast_pipeline
from src.forecasting.future_forecast import run_future_forecast

from src.utils.config_loader import load_config
from src.utils.io import save_csv
from src.utils.logger import get_logger
from src.utils.paths import (
    BEST_MODELS_PATH,
    COVID_DUMMY_PATH,
    COVID_EXCLUDED_PATH,
    COVID_FULL_SAMPLE_PATH,
    FORECAST_1Q_PATH,
    FORECAST_4Q_PATH,
    FORECAST_ALL_PATH,
    FUTURE_FORECAST_PATH,
    METRICS_1Q_PATH,
    METRICS_4Q_PATH,
    METRICS_ALL_PATH,
    MODEL_RANKING_PATH,
    OVERVIEW_SUMMARY_PATH,
    QUARTERLY_MODEL_INPUT_PATH,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    VARIABLE_METADATA_PATH,
)

logger = get_logger("run")


def _call_with_optional_config(func, *args, config=None):
    """
    Call a function that may or may not accept `config`.
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    required_positional = [
        p for p in params
        if p.default is inspect._empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]

    if config is None:
        return func(*args)

    if len(required_positional) >= len(args) + 1:
        return func(*args, config)

    return func(*args)


def _build_variable_metadata(config: dict) -> pd.DataFrame:
    rows = []
    for var_name, meta in config["data"]["variables"].items():
        rows.append(
            {
                "variable": var_name,
                "display_name": meta.get("display_name", var_name),
                "raw_frequency": meta.get("raw_frequency", ""),
                "target_type": meta.get("target_type", ""),
                "quarterly_method": meta.get("quarterly_method", ""),
            }
        )
    return pd.DataFrame(rows)


def _build_metrics_from_forecasts(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a standard metrics table directly from forecast outputs.
    Required columns:
    - variable
    - horizon
    - model
    - actual
    - forecast
    """
    required_cols = {"variable", "horizon", "model", "actual", "forecast"}
    if not required_cols.issubset(forecast_df.columns):
        missing = required_cols - set(forecast_df.columns)
        raise ValueError(
            f"forecast_df is missing required columns for metrics aggregation: {missing}"
        )

    work = forecast_df.copy()
    work["actual"] = pd.to_numeric(work["actual"], errors="coerce")
    work["forecast"] = pd.to_numeric(work["forecast"], errors="coerce")
    work = work.dropna(subset=["actual", "forecast"]).copy()

    work["error"] = work["forecast"] - work["actual"]
    work["abs_error"] = work["error"].abs()
    work["sq_error"] = work["error"] ** 2

    metrics_df = (
        work.groupby(["variable", "horizon", "model"], dropna=False)
        .agg(
            n_obs=("error", "size"),
            bias=("error", "mean"),
            mae=("abs_error", "mean"),
            rmse=("sq_error", lambda x: (x.mean()) ** 0.5),
        )
        .reset_index()
    )

    return metrics_df


def _split_metrics_by_horizon(metrics_df: pd.DataFrame):
    if metrics_df.empty or "horizon" not in metrics_df.columns:
        return metrics_df, pd.DataFrame(), pd.DataFrame()

    metrics_1q_df = metrics_df[metrics_df["horizon"] == 1].copy()
    metrics_4q_df = metrics_df[metrics_df["horizon"] == 4].copy()
    return metrics_df, metrics_1q_df, metrics_4q_df


def main() -> None:
    logger.info("Starting macro forecasting pipeline...")

    config = load_config()
    logger.info("Config loaded successfully.")
    logger.info("Project configuration summary:\n%s", pprint.pformat(config))

    # 1. Download raw data
    logger.info("Downloading raw data...")
    download_inflation_main()
    download_cash_rate_main()
    download_output_main()
    download_unemployment_main()
    download_wages_main()

    # 2. Process raw data into curated datasets
    logger.info("Processing curated datasets...")
    process_inflation_main()
    process_cash_rate_main()
    process_output_main()
    process_unemployment_main()
    process_wages_main()

    # 3. Load curated datasets
    logger.info("Loading curated datasets...")
    curated_data = load_all_curated_data()

    # 4. Resample each dataset first, then merge
    logger.info("Resampling all datasets to quarterly frequency...")
    quarterly_data = _call_with_optional_config(
        resample_all_to_quarterly,
        curated_data,
        config=config,
    )

    logger.info("Merging quarterly datasets...")
    merged_df = merge_quarterly_datasets(quarterly_data)
    quarterly_df = merged_df.copy()

    # 5. Transform variables / build targets / add COVID flags
    logger.info("Transforming variables...")
    transformed_df = _call_with_optional_config(
        transform_variables,
        quarterly_df,
        config=config,
    )

    logger.info("Building forecast targets...")
    model_input_df = _call_with_optional_config(
        build_forecast_targets,
        transformed_df,
        config=config,
    )

    logger.info("Adding COVID flags...")
    model_input_df = _call_with_optional_config(
        add_covid_flags,
        model_input_df,
        config=config,
    )

    save_csv(model_input_df, QUARTERLY_MODEL_INPUT_PATH)
    logger.info("Saved model input to %s", QUARTERLY_MODEL_INPUT_PATH)

    # 6. Optional train/test split
    logger.info("Creating train/test split...")
    train_df, test_df = _call_with_optional_config(
        split_train_test,
        model_input_df,
        config=config,
    )
    save_csv(train_df, TRAIN_DATA_PATH)
    save_csv(test_df, TEST_DATA_PATH)

    # 7. Rolling historical forecast evaluation
    logger.info("Running rolling forecast pipeline...")
    forecast_df = run_forecast_pipeline(model_input_df, config)

    save_csv(forecast_df, FORECAST_ALL_PATH)
    logger.info("Saved all rolling forecast results to %s", FORECAST_ALL_PATH)

    if not forecast_df.empty and "horizon" in forecast_df.columns:
        save_csv(forecast_df[forecast_df["horizon"] == 1], FORECAST_1Q_PATH)
        save_csv(forecast_df[forecast_df["horizon"] == 4], FORECAST_4Q_PATH)

    # 8. True future forecast
    logger.info("Running future forecast...")
    future_forecast_df = run_future_forecast(model_input_df, config)
    save_csv(future_forecast_df, FUTURE_FORECAST_PATH)
    logger.info("Saved future forecast to %s", FUTURE_FORECAST_PATH)

    # 9. Metrics
    logger.info("Computing metrics...")

    metrics_result = _call_with_optional_config(
        compute_metrics,
        forecast_df,
        config=config,
    )

    metrics_all_df = pd.DataFrame()

    if isinstance(metrics_result, pd.DataFrame):
        metrics_all_df = metrics_result.copy()
    elif isinstance(metrics_result, dict):
        if "metrics_all" in metrics_result and isinstance(metrics_result["metrics_all"], pd.DataFrame):
            metrics_all_df = metrics_result["metrics_all"].copy()

    required_metric_cols = {"variable", "horizon", "model", "rmse"}
    if metrics_all_df.empty or not required_metric_cols.issubset(metrics_all_df.columns):
        logger.warning(
            "compute_metrics() output is not directly usable. Rebuilding metrics from forecast_df."
        )
        metrics_all_df = _build_metrics_from_forecasts(forecast_df)

    metrics_all_df, metrics_1q_df, metrics_4q_df = _split_metrics_by_horizon(metrics_all_df)

    if not metrics_all_df.empty:
        save_csv(metrics_all_df, METRICS_ALL_PATH)
        logger.info("Saved all metrics to %s", METRICS_ALL_PATH)

    if not metrics_1q_df.empty:
        save_csv(metrics_1q_df, METRICS_1Q_PATH)

    if not metrics_4q_df.empty:
        save_csv(metrics_4q_df, METRICS_4Q_PATH)

    # 10. Summary and ranking
    logger.info("Summarizing outputs...")

    overview_summary_df = _call_with_optional_config(
        summarize_metrics,
        forecast_df,
        config=config,
    )

    model_ranking_df = _call_with_optional_config(
        rank_models,
        metrics_all_df,
        config=config,
    )

    if isinstance(overview_summary_df, pd.DataFrame):
        save_csv(overview_summary_df, OVERVIEW_SUMMARY_PATH)

    if isinstance(model_ranking_df, pd.DataFrame):
        save_csv(model_ranking_df, MODEL_RANKING_PATH)
        save_csv(model_ranking_df, BEST_MODELS_PATH)

    # 11. Variable metadata
    logger.info("Saving variable metadata...")
    variable_metadata_df = _build_variable_metadata(config)
    save_csv(variable_metadata_df, VARIABLE_METADATA_PATH)

    # 12. Robustness outputs
    logger.info("Building robustness outputs...")

    covid_cols = ["date"]
    if "is_covid_period" in model_input_df.columns:
        covid_cols.append("is_covid_period")
    if "is_post_covid_period" in model_input_df.columns:
        covid_cols.append("is_post_covid_period")

    forecast_with_flags = forecast_df.merge(
        model_input_df[covid_cols].drop_duplicates(subset=["date"]),
        on="date",
        how="left",
    )

    if "is_covid_period" not in forecast_with_flags.columns:
        forecast_with_flags["is_covid_period"] = 0

    if "is_post_covid_period" not in forecast_with_flags.columns:
        forecast_with_flags["is_post_covid_period"] = 0

    robustness_outputs = build_robustness_outputs(forecast_with_flags)

    if isinstance(robustness_outputs, dict):
        if "covid_full_sample_metrics" in robustness_outputs:
            save_csv(robustness_outputs["covid_full_sample_metrics"], COVID_FULL_SAMPLE_PATH)

        if "covid_excluded_metrics" in robustness_outputs:
            save_csv(robustness_outputs["covid_excluded_metrics"], COVID_EXCLUDED_PATH)

        if "covid_dummy_metrics" in robustness_outputs:
            save_csv(robustness_outputs["covid_dummy_metrics"], COVID_DUMMY_PATH)

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()