import streamlit as st

from src.dashboard.charts import metric_bar_chart
from src.dashboard.filters import filter_metrics
from src.dashboard.load_outputs import load_metrics
from src.dashboard.tables import prepare_metric_table

st.title("Model Performance")

try:
    metrics = load_metrics()
except FileNotFoundError:
    st.warning("Metric outputs not found. Run `python run.py` first.")
    st.stop()

variable = st.selectbox("Variable", sorted(metrics["variable"].unique()))
horizon = st.selectbox("Horizon", sorted(metrics["horizon"].unique()))
metric = st.selectbox("Metric", ["rmse", "bias", "mae"])

sub = filter_metrics(metrics, variable=variable, horizon=horizon)
st.plotly_chart(metric_bar_chart(sub, metric), use_container_width=True)
st.dataframe(prepare_metric_table(sub), use_container_width=True)