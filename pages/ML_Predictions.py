import streamlit as st
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
df = pd.read_csv("data/Finance_data.csv")

st.title("🤖 Investment Recommendation System")

# Encode Dataset
df_encoded = df.copy()

encoders = {}

for col in df_encoded.columns:

    if df_encoded[col].dtype == "object":

        le = LabelEncoder()

        df_encoded[col] = le.fit_transform(
            df_encoded[col].astype(str)
        )

        encoders[col] = le

# Target
target = "Avenue"

X = df_encoded.drop(target, axis=1)
y = df_encoded[target]

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

st.subheader("Enter Investor Details")

# User Inputs
age = st.number_input(
    "Age",
    min_value=18,
    max_value=80,
    value=25
)

gender = st.selectbox(
    "Gender",
    df["gender"].unique()
)

# Build Input Record
sample = {}

for col in X.columns:

    if col == "age":

        sample[col] = age

    elif col == "gender":

        sample[col] = encoders["gender"].transform(
            [gender]
        )[0]

    else:

        sample[col] = X[col].median()

# Prediction
if st.button("Predict Investment Avenue"):

    sample_df = pd.DataFrame(
        [sample]
    )

    prediction = model.predict(
        sample_df
    )[0]

    avenue = encoders["Avenue"].inverse_transform(
        [prediction]
    )[0]

    st.success(
        f"Recommended Investment Avenue: {avenue}"
    )
