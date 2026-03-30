import streamlit as st

from src.dashboard.load_outputs import load_model_input

st.title("Data Explorer")

try:
    df = load_model_input()
except FileNotFoundError:
    st.warning("Processed model input not found. Run `python run.py` first.")
    st.stop()

variables = [c for c in df.columns if c in ["output", "inflation", "cash_rate", "unemployment", "wages"]]
selected = st.selectbox("Variable", variables)

st.line_chart(df.set_index("date")[[selected]])
st.subheader("Data Preview")
st.dataframe(df[["date", selected]].tail(20), use_container_width=True)