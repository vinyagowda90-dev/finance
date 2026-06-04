import streamlit as st

from utils.data_loader import load_data
from utils.insights import get_kpis

df = load_data()

kpi = get_kpis(df)

st.title("📊 Dashboard")

c1,c2,c3,c4 = st.columns(4)

c1.metric("Investors",kpi["Total Investors"])
c2.metric("Avg Age",kpi["Average Age"])
c3.metric("Male",kpi["Male Investors"])
c4.metric("Female",kpi["Female Investors"])

st.dataframe(df.head())
