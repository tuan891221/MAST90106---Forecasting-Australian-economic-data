import streamlit as st
import plotly.express as px

from src.dashboard.load_outputs import load_model_input

st.title("Data Explorer")

df = load_model_input()

variables = ["output", "inflation", "cash_rate", "unemployment", "wages"]
variable = st.selectbox("Variable", variables, index=0)

plot_df = df[["date", variable]].dropna().copy()

fig = px.line(
    plot_df,
    x="date",
    y=variable,
    title=f"{variable} over time",
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Data Preview")
st.dataframe(plot_df.tail(15), use_container_width=True, hide_index=True)