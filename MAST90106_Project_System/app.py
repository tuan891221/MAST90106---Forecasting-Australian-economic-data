import streamlit as st
from pathlib import Path
import pandas as pd

from src.utils.paths import (
    QUARTERLY_MODEL_INPUT_PATH,
    FORECAST_ALL_PATH,
    METRICS_ALL_PATH,
    OVERVIEW_SUMMARY_PATH,
    BEST_MODELS_PATH,
)

st.set_page_config(
    page_title="Macro Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Australian Macroeconomic Forecasting Dashboard")
st.markdown(
    """
This dashboard presents a forecasting system for key Australian macroeconomic variables.

It combines:
- automated data processing,
- multiple forecasting models,
- forecast evaluation,
- and interactive visualisation through Streamlit.
"""
)

st.subheader("Project Scope")
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
**Variables**
- Real output growth
- Underlying inflation
- Cash rate
- Unemployment rate
- Wages growth

**Forecast Horizons**
- 1 quarter ahead
- 4 quarters ahead
"""
    )

with col2:
    st.markdown(
        """
**Models**
- Naive
- Historical Mean
- AR
- VAR
- BVAR
- Factor Model

**Evaluation Metrics**
- Bias
- RMSE
- MAE
"""
    )

st.subheader("System Status")

data_ready = QUARTERLY_MODEL_INPUT_PATH.exists()
forecast_ready = FORECAST_ALL_PATH.exists()
metrics_ready = METRICS_ALL_PATH.exists()

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Processed Data", "Ready" if data_ready else "Missing")
with c2:
    st.metric("Forecast Outputs", "Ready" if forecast_ready else "Missing")
with c3:
    st.metric("Metric Outputs", "Ready" if metrics_ready else "Missing")

if not (data_ready and forecast_ready and metrics_ready):
    st.warning("Some pipeline outputs are missing. Run `python run.py` first.")
else:
    st.success("All main outputs are available. You can explore the dashboard from the sidebar.")

st.subheader("Quick Start")
st.markdown(
    """
Use the sidebar to navigate:

- **Overview** → summary of results
- **Data Explorer** → inspect processed time series
- **Forecast Comparison** → compare forecasts against actual values
- **Model Performance** → compare RMSE / Bias / MAE
- **Heatmap Summary** → overall model ranking view
- **Raw Results** → inspect output tables
"""
)

if OVERVIEW_SUMMARY_PATH.exists():
    st.subheader("Pipeline Summary")
    try:
        summary_df = pd.read_csv(OVERVIEW_SUMMARY_PATH)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.info(f"Overview summary exists but could not be displayed: {e}")

if BEST_MODELS_PATH.exists():
    st.subheader("Best Models Snapshot")
    try:
        best_df = pd.read_csv(BEST_MODELS_PATH)
        st.dataframe(best_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.info(f"Best model summary exists but could not be displayed: {e}")

st.markdown("---")
st.caption("Built with Streamlit for the Forecasting Australian economic data project.")
st.caption("All rights reserved by Team 37")