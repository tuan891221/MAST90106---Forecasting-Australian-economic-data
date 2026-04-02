import streamlit as st

from src.dashboard.load_outputs import load_metrics
from src.dashboard.charts import metric_bar_chart

st.title("Model Performance")

metrics_df = load_metrics()

metric = st.selectbox("Metric", ["rmse", "mae", "bias"], index=0)

fig = metric_bar_chart(metrics_df, metric)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Metrics Table")
st.dataframe(metrics_df, use_container_width=True, hide_index=True)