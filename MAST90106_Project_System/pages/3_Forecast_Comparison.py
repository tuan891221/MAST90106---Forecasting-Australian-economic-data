import streamlit as st

from src.dashboard.load_outputs import load_single_variable_history_and_forecast
from src.dashboard.charts import single_variable_history_forecast_chart

st.title("Forecast Comparison")

variable = st.selectbox(
    "Variable",
    ["output", "inflation", "cash_rate", "unemployment", "wages"],
    index=0,
)

model = st.selectbox(
    "Model",
    ["naive", "mean", "ar", "var", "bvar", "factor"],
    index=4,
)

horizon = st.selectbox(
    "Rolling Forecast Horizon",
    [1, 4],
    index=0,
)

hist_df, roll_df, future_df = load_single_variable_history_and_forecast(
    variable=variable,
    model=model,
    horizon=horizon,
)

fig = single_variable_history_forecast_chart(
    hist_df=hist_df,
    roll_df=roll_df,
    future_df=future_df,
    variable=variable,
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Historical Data")
st.dataframe(hist_df.tail(12), use_container_width=True, hide_index=True)

st.subheader("Rolling Forecast")
st.dataframe(roll_df.tail(12), use_container_width=True, hide_index=True)

st.subheader("Future Forecast Path")
st.dataframe(future_df, use_container_width=True, hide_index=True)