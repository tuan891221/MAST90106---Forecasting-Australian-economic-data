import streamlit as st

from src.dashboard.charts import heatmap_chart
from src.dashboard.load_outputs import load_metrics

st.title("Heatmap Summary")

try:
    metrics = load_metrics()
except FileNotFoundError:
    st.warning("Metric outputs not found. Run `python run.py` first.")
    st.stop()

metric = st.selectbox("Metric", ["rmse", "bias", "mae"])
horizon = st.selectbox("Horizon", sorted(metrics["horizon"].unique()))

st.plotly_chart(heatmap_chart(metrics, metric, horizon), use_container_width=True)