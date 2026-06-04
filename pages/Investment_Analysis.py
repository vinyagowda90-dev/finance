import streamlit as st
import plotly.express as px

from utils.data_loader import load_data

df = load_data()

st.title("📈 Investment Analysis")

avenue = df["Avenue"].value_counts()

fig = px.bar(
    x=avenue.index,
    y=avenue.values,
    labels={
        "x":"Investment Type",
        "y":"Investors"
    }
)

st.plotly_chart(
    fig,
    use_container_width=True
)
