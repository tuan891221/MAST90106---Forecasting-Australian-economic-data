import streamlit as st

from src.dashboard.load_outputs import load_combined_history_and_forecast
from src.dashboard.charts import combined_history_forecast_chart

st.title("Combined Forecast View")

model = st.selectbox(
    "Model",
    ["naive", "mean", "ar", "var", "bvar", "factor"],
    index=4,
)

hist_df, future_df = load_combined_history_and_forecast(model=model)

if future_df.empty:
    st.warning("No future forecast results found for this model.")
    st.stop()

forecast_start_date = future_df["date"].min()

fig = combined_history_forecast_chart(
    historical_df=hist_df,
    forecast_df=future_df,
    forecast_start_date=forecast_start_date,
)

st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
**How to read this chart**
- Solid lines = historical values
- Dashed lines = future forecasts
- Values are standardized (z-score) so that all variables can be plotted on one axis
"""
)

st.subheader("Future Forecast Table")
st.dataframe(
    future_df.sort_values(["variable", "date"]),
    use_container_width=True,
    hide_index=True,
)