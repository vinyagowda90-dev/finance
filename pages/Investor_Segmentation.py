import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from utils.data_loader import load_data

st.set_page_config(
    page_title="Investor Segmentation",
    layout="wide"
)

st.title("👥 Investor Segmentation")

# -------------------------
# Load Data
# -------------------------

df = load_data()

# -------------------------
# Features for Clustering
# -------------------------

features = [
    "Mutual_Funds",
    "Equity_Market",
    "Debentures",
    "Government_Bonds",
    "Fixed_Deposits",
    "PPF",
    "Gold"
]

available_features = [
    col for col in features
    if col in df.columns
]

cluster_df = df[available_features]

# -------------------------
# Scale Data
# -------------------------

scaler = StandardScaler()

scaled_data = scaler.fit_transform(
    cluster_df
)

# -------------------------
# Number of Clusters
# -------------------------

st.sidebar.header("Clustering Settings")

n_clusters = st.sidebar.slider(
    "Number of Clusters",
    min_value=2,
    max_value=6,
    value=3
)

# -------------------------
# KMeans Model
# -------------------------

kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(
    scaled_data
)

df["Cluster"] = clusters

# -------------------------
# Cluster Statistics
# -------------------------

st.subheader("📊 Cluster Summary")

cluster_summary = (
    df.groupby("Cluster")[available_features]
      .mean()
      .round(2)
)

st.dataframe(cluster_summary)

# -------------------------
# Cluster Counts
# -------------------------

st.subheader("📈 Cluster Distribution")

cluster_count = (
    df["Cluster"]
    .value_counts()
    .reset_index()
)

cluster_count.columns = [
    "Cluster",
    "Count"
]

fig = px.bar(
    cluster_count,
    x="Cluster",
    y="Count",
    text="Count",
    title="Investors per Cluster"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -------------------------
# PCA Visualization
# -------------------------

st.subheader("🧠 PCA Cluster Visualization")

pca = PCA(
    n_components=2
)

pca_data = pca.fit_transform(
    scaled_data
)

plot_df = pd.DataFrame({
    "PCA1": pca_data[:, 0],
    "PCA2": pca_data[:, 1],
    "Cluster": clusters.astype(str)
})

fig = px.scatter(
    plot_df,
    x="PCA1",
    y="PCA2",
    color="Cluster",
    size_max=20,
    title="Investor Segments"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -------------------------
# 3D Cluster Visualization
# -------------------------

st.subheader("🚀 3D Cluster Visualization")

pca3 = PCA(
    n_components=3
)

pca3_data = pca3.fit_transform(
    scaled_data
)

plot3d = pd.DataFrame({
    "PC1": pca3_data[:, 0],
    "PC2": pca3_data[:, 1],
    "PC3": pca3_data[:, 2],
    "Cluster": clusters.astype(str)
})

fig = px.scatter_3d(
    plot3d,
    x="PC1",
    y="PC2",
    z="PC3",
    color="Cluster",
    title="3D Investor Segmentation"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -------------------------
# Investor Personas
# -------------------------

st.subheader("🎯 Investor Personas")

for cluster in sorted(df["Cluster"].unique()):

    st.markdown(
        f"### Cluster {cluster}"
    )

    avg = (
        df[df["Cluster"] == cluster]
        [available_features]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    top_choice = avg.index[0]

    if top_choice in [
        "Fixed_Deposits",
        "Government_Bonds",
        "PPF"
    ]:

        persona = "🛡️ Conservative Investor"

    elif top_choice in [
        "Mutual_Funds",
        "Gold"
    ]:

        persona = "⚖️ Moderate Investor"

    else:

        persona = "🔥 Aggressive Investor"

    st.success(persona)

    st.write(
        "Top Preferred Investment:",
        top_choice
    )

# -------------------------
# Download Results
# -------------------------

st.subheader("⬇️ Download Clustered Data")

csv = df.to_csv(
    index=False
)

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="investor_segments.csv",
    mime="text/csv"
)
