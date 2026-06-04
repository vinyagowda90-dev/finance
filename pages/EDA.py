import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data

st.set_page_config(layout="wide")

df = load_data()

st.title("📊 Exploratory Data Analysis")

# -------------------------
# Dataset Overview
# -------------------------

st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Missing Values", df.isnull().sum().sum())

st.dataframe(df.head())

# -------------------------
# Data Types
# -------------------------

st.subheader("Column Information")

info_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str)
})

st.dataframe(info_df)

# -------------------------
# Numerical Summary
# -------------------------

st.subheader("Statistical Summary")

st.dataframe(df.describe())

# -------------------------
# Gender Analysis
# -------------------------

if "gender" in df.columns:

    st.subheader("👥 Gender Distribution")

    gender_count = (
        df["gender"]
        .value_counts()
        .reset_index()
    )

    gender_count.columns = ["Gender", "Count"]

    fig = px.pie(
        gender_count,
        names="Gender",
        values="Count",
        hole=0.4
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# -------------------------
# Age Analysis
# -------------------------

if "age" in df.columns:

    st.subheader("🎂 Age Distribution")

    fig = px.histogram(
        df,
        x="age",
        nbins=10,
        marginal="box"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# -------------------------
# Investment Avenue
# -------------------------

if "Avenue" in df.columns:

    st.subheader("💰 Preferred Investment Avenue")

    avenue = (
        df["Avenue"]
        .value_counts()
        .reset_index()
    )

    avenue.columns = ["Avenue", "Count"]

    fig = px.bar(
        avenue,
        x="Avenue",
        y="Count",
        text="Count"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# -------------------------
# Objective Analysis
# -------------------------

if "Objective" in df.columns:

    st.subheader("🎯 Investment Objective")

    objective = (
        df["Objective"]
        .value_counts()
        .reset_index()
    )

    objective.columns = ["Objective", "Count"]

    fig = px.bar(
        objective,
        x="Objective",
        y="Count",
        text="Count"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# -------------------------
# Duration Analysis
# -------------------------

if "Duration" in df.columns:

    st.subheader("⏳ Investment Duration")

    duration = (
        df["Duration"]
        .value_counts()
        .reset_index()
    )

    duration.columns = ["Duration", "Count"]

    fig = px.bar(
        duration,
        x="Duration",
        y="Count",
        text="Count"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# -------------------------
# Expected Returns
# -------------------------

if "Expect" in df.columns:

    st.subheader("📈 Expected Returns")

    returns = (
        df["Expect"]
        .value_counts()
        .reset_index()
    )

    returns.columns = ["Expected Return", "Count"]

    fig = px.pie(
        returns,
        names="Expected Return",
        values="Count"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# -------------------------
# Information Source
# -------------------------

if "Source" in df.columns:

    st.subheader("🌐 Information Source")

    source = (
        df["Source"]
        .value_counts()
        .reset_index()
    )

    source.columns = ["Source", "Count"]

    fig = px.bar(
        source,
        x="Source",
        y="Count",
        text="Count"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# -------------------------
# Correlation Heatmap
# -------------------------

st.subheader("🔥 Correlation Heatmap")

numeric_df = df.select_dtypes(
    include=["int64", "float64"]
)

if len(numeric_df.columns) > 1:

    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# -------------------------
# Key Insights
# -------------------------

st.subheader("🧠 Key Insights")

st.info(
    f"""
    • Total Investors Surveyed: {len(df)}

    • Most Popular Investment:
      {df['Avenue'].mode()[0] if 'Avenue' in df.columns else 'N/A'}

    • Most Common Objective:
      {df['Objective'].mode()[0] if 'Objective' in df.columns else 'N/A'}

    • Average Age:
      {round(df['age'].mean(),1) if 'age' in df.columns else 'N/A'}
    """
)
