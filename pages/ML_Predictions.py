import streamlit as st
import pandas as pd

from utils.model_utils import predict_all

st.title("🤖 Investment Predictor")

age = st.slider(
    "Age",
    18,
    60,
    25
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

objective = st.selectbox(
    "Objective",
    [
        "Capital Appreciation",
        "Growth",
        "Income"
    ]
)

duration = st.selectbox(
    "Duration",
    [
        "1-3 years",
        "3-5 years",
        "5+ years"
    ]
)

if st.button("Predict"):

    sample = pd.DataFrame({
        "age":[age]
    })

    prediction = predict_all(sample)

    st.subheader(
        "Model Predictions"
    )

    st.write(prediction)
