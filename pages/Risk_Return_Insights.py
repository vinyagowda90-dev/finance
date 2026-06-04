import streamlit as st
import plotly.express as px

from utils.data_loader import load_data

df = load_data()

st.title("🎯 Return Expectations")

fig = px.pie(
    df,
    names="Expect"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

fig2 = px.bar(
    df["Factor"].value_counts().reset_index(),
    x="Factor",
    y="count"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)
