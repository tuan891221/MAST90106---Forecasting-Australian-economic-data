VARIABLES = ["output", "inflation", "cash_rate", "unemployment", "wages"]
MODELS = ["naive", "mean", "ar", "var", "bvar", "factor"]
HORIZONS = [1, 4]
METRICS = ["bias", "rmse", "mae"]

DISPLAY_NAMES = {
    "output": "Real Non-farm Output Growth",
    "inflation": "Underlying Inflation",
    "cash_rate": "Cash Rate",
    "unemployment": "Unemployment Rate",
    "wages": "Wages Growth",
    "naive": "Naive",
    "mean": "Historical Mean",
    "ar": "AR",
    "var": "VAR",
    "bvar": "BVAR",
    "factor": "Factor Model",
}