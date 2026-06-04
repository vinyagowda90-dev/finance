import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data

df = load_data()

st.title("🔥 Advanced Analytics")

numeric = [
    "Mutual_Funds",
    "Equity_Market",
    "Debentures",
    "Government_Bonds",
    "Fixed_Deposits",
    "PPF",
    "Gold"
]

corr = df[numeric].corr()

fig = px.imshow(
    corr,
    text_auto=True,
    aspect="auto",
    title="Investment Correlation Heatmap"
)

st.plotly_chart(
    fig,
    use_container_width=True
)
