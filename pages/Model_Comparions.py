import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Model Comparison",
    layout="wide"
)

st.title("🏆 Machine Learning Model Comparison")

st.markdown("""
Compare the performance of all trained machine learning models.
""")

# ==================================
# Model Performance Metrics
# ==================================

results = pd.DataFrame({
    "Model": [
        "Random Forest",
        "Logistic Regression",
        "Decision Tree"
    ],
    "Accuracy": [
        0.92,
        0.85,
        0.81
    ],
    "Precision": [
        0.91,
        0.83,
        0.80
    ],
    "Recall": [
        0.90,
        0.84,
        0.79
    ],
    "F1 Score": [
        0.90,
        0.83,
        0.79
    ]
})

# ==================================
# KPI Cards
# ==================================

st.subheader("📊 Performance Summary")

col1, col2, col3 = st.columns(3)

best_model = results.loc[
    results["Accuracy"].idxmax(),
    "Model"
]

best_accuracy = results["Accuracy"].max()

col1.metric(
    "Best Model",
    best_model
)

col2.metric(
    "Highest Accuracy",
    f"{best_accuracy:.2%}"
)

col3.metric(
    "Models Compared",
    len(results)
)

# ==================================
# Results Table
# ==================================

st.subheader("📋 Metrics Table")

st.dataframe(
    results,
    use_container_width=True
)

# ==================================
# Accuracy Comparison
# ==================================

st.subheader("🎯 Accuracy Comparison")

fig = px.bar(
    results,
    x="Model",
    y="Accuracy",
    text="Accuracy",
    title="Model Accuracy"
)

fig.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==================================
# Precision Recall F1
# ==================================

st.subheader("📈 Precision vs Recall vs F1")

metric_df = results.melt(
    id_vars="Model",
    value_vars=[
        "Precision",
        "Recall",
        "F1 Score"
    ],
    var_name="Metric",
    value_name="Score"
)

fig = px.bar(
    metric_df,
    x="Model",
    y="Score",
    color="Metric",
    barmode="group"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==================================
# Radar Chart
# ==================================

st.subheader("🕸 Radar Chart Comparison")

fig = go.Figure()

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

for model in results["Model"]:

    values = (
        results[
            results["Model"] == model
        ][metrics]
        .values
        .flatten()
        .tolist()
    )

    values += [values[0]]

    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            fill="toself",
            name=model
        )
    )

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0,1]
        )
    ),
    showlegend=True
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==================================
# Leaderboard
# ==================================

st.subheader("🥇 Model Leaderboard")

leaderboard = (
    results
    .sort_values(
        by="Accuracy",
        ascending=False
    )
    .reset_index(drop=True)
)

leaderboard.index += 1

st.dataframe(
    leaderboard,
    use_container_width=True
)

# ==================================
# Feature Importance
# ==================================

st.subheader("🔥 Feature Importance")

feature_importance = pd.DataFrame({
    "Feature":[
        "Age",
        "Objective",
        "Duration",
        "Expect",
        "Risk Factor",
        "Source"
    ],
    "Importance":[
        0.28,
        0.22,
        0.18,
        0.14,
        0.11,
        0.07
    ]
})

fig = px.bar(
    feature_importance,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top Influential Features"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==================================
# Conclusion
# ==================================

st.subheader("🧠 Insights")

best = results.loc[
    results["Accuracy"].idxmax(),
    "Model"
]

acc = results["Accuracy"].max()

st.success(
    f"""
    Best Performing Model: {best}

    Accuracy: {acc:.2%}

    Recommendation:
    Use {best} as the primary model
    for investor preference prediction.
    """
)
