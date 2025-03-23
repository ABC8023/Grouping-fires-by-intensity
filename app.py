import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Load the cleaned + scaled dataset
@st.cache_data
def load_data():
    df = pd.read_csv("fire_scaled.csv")  # Preprocessed data with GMM cluster
    return df

data = load_data()

# Sidebar options
st.sidebar.title("🔥 Fire Intensity Clustering Dashboard")
page = st.sidebar.selectbox("Select Section", ["Data Overview", "PCA Visualization", "Model Metrics", "Cluster Analysis"])

# Data Overview Page
if page == "Data Overview":
    st.title("📊 Fire Dataset Overview")
    st.dataframe(data.head())
    st.write("Shape:", data.shape)
    st.write("Column Descriptions:")
    st.json({col: str(dtype) for col, dtype in data.dtypes.items()})

# PCA Visualization
elif page == "PCA Visualization":
    st.title("🎯 PCA Visualization of GMM Clusters")

    pca = PCA(n_components=2)
    components = pca.fit_transform(data[["log_brightness", "log_bright_t31", "log_frp", "confidence"]])
    data["PCA1"] = components[:, 0]
    data["PCA2"] = components[:, 1]

    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x="PCA1", y="PCA2", hue="gmm_cluster", palette="Set2", ax=ax)
    st.pyplot(fig)

# Model Metrics
elif page == "Model Metrics":
    st.title("📈 Clustering Model Performance")
    
    silhouette = silhouette_score(data[["log_brightness", "log_bright_t31", "log_frp", "confidence"]], data["gmm_cluster"])
    db_score = davies_bouldin_score(data[["log_brightness", "log_bright_t31", "log_frp", "confidence"]], data["gmm_cluster"])
    
    st.metric("Silhouette Score", f"{silhouette:.4f}")
    st.metric("Davies-Bouldin Index", f"{db_score:.4f}")
    
    st.write("📌 These scores help determine cluster quality.")

# Cluster Analysis
elif page == "Cluster Analysis":
    st.title("🔍 Cluster-Based Feature Averages")

    cluster_summary = data.groupby("gmm_cluster")[["brightness", "bright_t31", "frp", "confidence"]].mean().reset_index()
    st.dataframe(cluster_summary)

    fig, ax = plt.subplots()
    cluster_summary.set_index("gmm_cluster").plot(kind="bar", ax=ax)
    plt.title("Average Feature Values per GMM Cluster")
    st.pyplot(fig)