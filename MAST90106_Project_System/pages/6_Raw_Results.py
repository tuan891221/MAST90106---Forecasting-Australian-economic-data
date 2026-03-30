import streamlit as st

from src.dashboard.load_outputs import load_forecasts, load_metrics
from src.dashboard.tables import prepare_forecast_table, prepare_metric_table

st.title("Raw Results")

tab1, tab2 = st.tabs(["Forecasts", "Metrics"])

with tab1:
    try:
        forecasts = load_forecasts()
        st.dataframe(prepare_forecast_table(forecasts), use_container_width=True)
    except FileNotFoundError:
        st.warning("Forecast outputs not found.")

with tab2:
    try:
        metrics = load_metrics()
        st.dataframe(prepare_metric_table(metrics), use_container_width=True)
    except FileNotFoundError:
        st.warning("Metric outputs not found.")