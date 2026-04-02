import streamlit as st
import pandas as pd

from src.utils.paths import OVERVIEW_SUMMARY_PATH, BEST_MODELS_PATH

st.title("Overview")

if OVERVIEW_SUMMARY_PATH.exists():
    summary = pd.read_csv(OVERVIEW_SUMMARY_PATH)
    st.subheader("Project Summary")
    st.dataframe(summary, use_container_width=True)
else:
    st.info("Run `python run.py` first to generate dashboard summary files.")

if BEST_MODELS_PATH.exists():
    best = pd.read_csv(BEST_MODELS_PATH)
    st.subheader("Best Models by Variable and Horizon")
    st.dataframe(best, use_container_width=True)