import streamlit as st

from src.dashboard.charts import forecast_chart
from src.dashboard.filters import filter_forecasts
from src.dashboard.load_outputs import load_forecasts

st.title("Forecast Comparison")

try:
    forecasts = load_forecasts()
except FileNotFoundError:
    st.warning("Forecast outputs not found. Run `python run.py` first.")
    st.stop()

variable = st.selectbox("Variable", sorted(forecasts["variable"].unique()))
horizon = st.selectbox("Horizon", sorted(forecasts["horizon"].unique()))
models = st.multiselect("Models", sorted(forecasts["model"].unique()), default=sorted(forecasts["model"].unique()))

sub = filter_forecasts(forecasts, variable, horizon, models)
st.plotly_chart(forecast_chart(sub, variable), use_container_width=True)
st.dataframe(sub.sort_values(["date", "model"]), use_container_width=True)