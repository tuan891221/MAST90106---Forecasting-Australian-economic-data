import streamlit as st

from src.dashboard.load_outputs import load_forecasts, load_metrics

st.title("Raw Results")

forecast_df = load_forecasts()
metrics_df = load_metrics()

st.subheader("Forecast Results")
st.dataframe(forecast_df, use_container_width=True, hide_index=True)

st.subheader("Metrics Results")
st.dataframe(metrics_df, use_container_width=True, hide_index=True)