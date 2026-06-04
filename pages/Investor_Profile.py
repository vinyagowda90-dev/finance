import streamlit as st

from utils.data_loader import load_data
from utils.charts import gender_chart,age_chart

df = load_data()

st.title("👥 Investor Profile")

st.plotly_chart(
    gender_chart(df),
    use_container_width=True
)

st.plotly_chart(
    age_chart(df),
    use_container_width=True
)
