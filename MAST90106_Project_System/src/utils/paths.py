from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "config.yaml"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_RBA_DIR = RAW_DIR / "rba"
RAW_ABS_DIR = RAW_DIR / "abs"
RAW_FRED_DIR = RAW_DIR / "fred"

CURATED_DIR = DATA_DIR / "curated"
CURATED_INFLATION_DIR = CURATED_DIR / "inflation"
CURATED_CASH_RATE_DIR = CURATED_DIR / "cash_rate"
CURATED_OUTPUT_DIR = CURATED_DIR / "output"
CURATED_UNEMPLOYMENT_DIR = CURATED_DIR / "unemployment"
CURATED_WAGES_DIR = CURATED_DIR / "wages"

PROCESSED_DIR = DATA_DIR / "processed"
QUARTERLY_MODEL_INPUT_PATH = PROCESSED_DIR / "quarterly_model_input.csv"
TRAIN_DATA_PATH = PROCESSED_DIR / "train_data.csv"
TEST_DATA_PATH = PROCESSED_DIR / "test_data.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FORECAST_DIR = OUTPUT_DIR / "forecasts"
METRICS_DIR = OUTPUT_DIR / "metrics"
DASHBOARD_DIR = OUTPUT_DIR / "dashboard"
ROBUSTNESS_DIR = OUTPUT_DIR / "robustness"
LOG_DIR = OUTPUT_DIR / "logs"

FORECAST_1Q_PATH = FORECAST_DIR / "forecast_1q.csv"
FORECAST_4Q_PATH = FORECAST_DIR / "forecast_4q.csv"
FORECAST_ALL_PATH = FORECAST_DIR / "forecast_all.csv"

METRICS_1Q_PATH = METRICS_DIR / "metrics_1q.csv"
METRICS_4Q_PATH = METRICS_DIR / "metrics_4q.csv"
METRICS_ALL_PATH = METRICS_DIR / "metrics_all.csv"
MODEL_RANKING_PATH = METRICS_DIR / "model_ranking.csv"

OVERVIEW_SUMMARY_PATH = DASHBOARD_DIR / "overview_summary.csv"
BEST_MODELS_PATH = DASHBOARD_DIR / "best_models.csv"
VARIABLE_METADATA_PATH = DASHBOARD_DIR / "variable_metadata.csv"

COVID_FULL_SAMPLE_PATH = ROBUSTNESS_DIR / "covid_full_sample_metrics.csv"
COVID_EXCLUDED_PATH = ROBUSTNESS_DIR / "covid_excluded_metrics.csv"
COVID_DUMMY_PATH = ROBUSTNESS_DIR / "covid_dummy_metrics.csv"

PIPELINE_LOG_PATH = LOG_DIR / "pipeline.log"


def ensure_directories() -> None:
    for d in [
        RAW_RBA_DIR,
        RAW_ABS_DIR,
        RAW_FRED_DIR,
        CURATED_INFLATION_DIR,
        CURATED_CASH_RATE_DIR,
        CURATED_OUTPUT_DIR,
        CURATED_UNEMPLOYMENT_DIR,
        CURATED_WAGES_DIR,
        PROCESSED_DIR,
        FORECAST_DIR,
        METRICS_DIR,
        DASHBOARD_DIR,
        ROBUSTNESS_DIR,
        LOG_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)