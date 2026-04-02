import streamlit as st

from src.dashboard.load_outputs import load_metrics
from src.dashboard.charts import heatmap_chart

st.title("Heatmap Summary")

metrics_df = load_metrics()

metric = st.selectbox("Metric", ["rmse", "mae", "bias"], index=0)

fig = heatmap_chart(metrics_df, metric=metric)
st.plotly_chart(fig, use_container_width=True)