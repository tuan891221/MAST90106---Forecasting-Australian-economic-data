from __future__ import annotations

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
from src.evaluation.rank_models import rank_models
from src.evaluation.robustness_checks import build_robustness_outputs
from src.evaluation.summarize_metrics import summarize_metrics
from src.forecasting.forecast_pipeline import run_forecast_pipeline
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
    METRICS_1Q_PATH,
    METRICS_4Q_PATH,
    METRICS_ALL_PATH,
    MODEL_RANKING_PATH,
    OVERVIEW_SUMMARY_PATH,
    QUARTERLY_MODEL_INPUT_PATH,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    VARIABLE_METADATA_PATH,
    ensure_directories,
)


def build_dashboard_outputs(model_input_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    summary = pd.DataFrame(
        {
            "item": [
                "n_rows",
                "start_date",
                "end_date",
                "n_variables",
                "n_metrics_rows",
            ],
            "value": [
                len(model_input_df),
                model_input_df["date"].min(),
                model_input_df["date"].max(),
                5,
                len(metrics_df),
            ],
        }
    )
    save_csv(summary, OVERVIEW_SUMMARY_PATH, index=False)

    variable_metadata = pd.DataFrame(
        {
            "variable": ["output", "inflation", "cash_rate", "unemployment", "wages"],
            "display_name": [
                "Real Non-farm Output Growth",
                "Underlying Inflation",
                "Cash Rate",
                "Unemployment Rate",
                "Wages Growth",
            ],
        }
    )
    save_csv(variable_metadata, VARIABLE_METADATA_PATH, index=False)

    best_models = (
        metrics_df.sort_values(["variable", "horizon", "rmse"])
        .groupby(["variable", "horizon"], as_index=False)
        .first()[["variable", "horizon", "model", "rmse"]]
    )
    save_csv(best_models, BEST_MODELS_PATH, index=False)


def main() -> None:
    logger = get_logger("run")
    logger.info("Starting macro forecasting pipeline...")

    ensure_directories()
    config = load_config()

    logger.info("Project configuration summary:")
    pprint.pprint(config)

    # 1. raw download
    logger.info("Downloading raw files where configured...")
    download_inflation_main()
    download_cash_rate_main()
    download_output_main()
    download_unemployment_main()
    download_wages_main()

    # 2. processors
    logger.info("Processing raw files into curated datasets...")
    process_inflation_main()
    process_cash_rate_main()
    process_output_main()
    process_unemployment_main()
    process_wages_main()

    # 3. data pipeline
    datasets = load_all_curated_data()
    quarterly_datasets = resample_all_to_quarterly(datasets)
    merged_df = merge_quarterly_datasets(quarterly_datasets)
    transformed_df = transform_variables(merged_df)
    model_input_df = build_forecast_targets(transformed_df)
    model_input_df = add_covid_flags(model_input_df)

    save_csv(model_input_df, QUARTERLY_MODEL_INPUT_PATH, index=False)

    train_df, test_df = split_train_test(model_input_df)
    save_csv(train_df, TRAIN_DATA_PATH, index=False)
    save_csv(test_df, TEST_DATA_PATH, index=False)

    # 4. forecasting
    forecast_df = run_forecast_pipeline(model_input_df, config)
    save_csv(forecast_df, FORECAST_ALL_PATH, index=False)
    save_csv(forecast_df[forecast_df["horizon"] == 1], FORECAST_1Q_PATH, index=False)
    save_csv(forecast_df[forecast_df["horizon"] == 4], FORECAST_4Q_PATH, index=False)

    # 5. evaluation
    metrics_df = summarize_metrics(forecast_df)
    save_csv(metrics_df, METRICS_ALL_PATH, index=False)
    save_csv(metrics_df[metrics_df["horizon"] == 1], METRICS_1Q_PATH, index=False)
    save_csv(metrics_df[metrics_df["horizon"] == 4], METRICS_4Q_PATH, index=False)

    ranking_df = rank_models(metrics_df)
    save_csv(ranking_df, MODEL_RANKING_PATH, index=False)

    # 6. robustness
    full_df, excluded_df, dummy_df = build_robustness_outputs(forecast_df)
    save_csv(full_df, COVID_FULL_SAMPLE_PATH, index=False)
    save_csv(excluded_df, COVID_EXCLUDED_PATH, index=False)
    save_csv(dummy_df, COVID_DUMMY_PATH, index=False)

    # 7. dashboard files
    build_dashboard_outputs(model_input_df, metrics_df)

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()