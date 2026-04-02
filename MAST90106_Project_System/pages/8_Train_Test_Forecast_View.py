import streamlit as st

from src.dashboard.load_outputs import load_quarterly_model_input
from src.dashboard.holdout_utils import build_holdout_forecast_view
from src.dashboard.charts import holdout_all_variables_chart
from src.utils.config_loader import load_config

st.title("Train / Test Forecast View")

config = load_config()
df = load_quarterly_model_input()

model = st.selectbox(
    "Model",
    ["naive", "mean", "ar", "var", "bvar", "factor"],
    index=4,
)

test_size = st.selectbox(
    "Test Size (quarters)",
    [4, 8, 12],
    index=1,
)

actual_df, fitted_df, forecast_df = build_holdout_forecast_view(
    df=df,
    config=config,
    model_name=model,
    test_size=test_size,
)

variables = ["output", "inflation", "cash_rate", "unemployment", "wages"]
variables = [v for v in variables if v in actual_df.columns]

fig = holdout_all_variables_chart(
    actual_df=actual_df,
    fitted_df=fitted_df,
    forecast_df=forecast_df,
    variables=variables,
)

st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
**How to read this page**
- Solid line = Actual
- Dotted line = Fitted values inside the training sample
- Dashed line = Forecast over the held-out test sample
- Vertical dotted line = start of the held-out test forecast period
"""
)

with st.expander("Actual data"):
    st.dataframe(actual_df.tail(20), use_container_width=True, hide_index=True)

with st.expander("Fitted (Train)"):
    st.dataframe(fitted_df.tail(20), use_container_width=True, hide_index=True)

with st.expander("Forecast (Test)"):
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)