import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="Fire Intensity Clustering", layout="wide")

# Load data
@st.cache_data

def load_data(uploaded_file=None):
    if uploaded_file:
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_csv("fire_scaled.csv")

st.sidebar.title("🔥 Fire Intensity Clustering Dashboard")
uploaded_file = st.sidebar.file_uploader("Upload your fire_scaled.csv", type="csv")
data = load_data(uploaded_file)

# Sidebar Navigation
page = st.sidebar.selectbox("Select Section", [
    "Data Overview", "Data Exploration", "PCA Visualization", "Interactive Charts", 
    "Model Metrics", "Feature Importance", "Clustering Interface", "Comparative Analysis"
])

# Data Overview
if page == "Data Overview":
    st.title("📊 Fire Dataset Overview")
    st.dataframe(data.head())
    st.write("Shape:", data.shape)
    st.write("Column Descriptions:")
    st.json({col: str(dtype) for col, dtype in data.dtypes.items()})

# Data Exploration
elif page == "Data Exploration":
    st.title("🔍 Interactive Data Exploration")
    selected_feature = st.selectbox("Select a feature to explore", data.columns)
    st.bar_chart(data[selected_feature].value_counts().sort_index())

# PCA Visualization
elif page == "PCA Visualization":
    st.title("🎯 PCA Visualization")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data[["log_brightness", "log_bright_t31", "log_frp", "confidence"]])
    data["PCA1"] = pca_result[:, 0]
    data["PCA2"] = pca_result[:, 1]

    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x="PCA1", y="PCA2", hue="gmm_cluster", palette="Set2", ax=ax)
    st.pyplot(fig)

# Interactive Charts
elif page == "Interactive Charts":
    st.title("📊 Interactive Feature Distributions")
    selected_col = st.selectbox("Choose a feature", ["brightness", "bright_t31", "frp", "confidence"])
    fig, ax = plt.subplots()
    sns.histplot(data[selected_col], kde=True, ax=ax)
    st.pyplot(fig)

# Model Metrics
elif page == "Model Metrics":
    st.title("📈 Clustering Model Performance")
    silhouette = silhouette_score(data[["log_brightness", "log_bright_t31", "log_frp", "confidence"]], data["gmm_cluster"])
    db_score = davies_bouldin_score(data[["log_brightness", "log_bright_t31", "log_frp", "confidence"]], data["gmm_cluster"])
    st.metric("Silhouette Score", f"{silhouette:.4f}")
    st.metric("Davies-Bouldin Index", f"{db_score:.4f}")

# Feature Importance (Correlation Heatmap)
elif page == "Feature Importance":
    st.title("🔥 Feature Correlation Heatmap")
    features = ["brightness", "bright_t31", "frp", "confidence"]
    fig, ax = plt.subplots()
    sns.heatmap(data[features].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Clustering Interface
elif page == "Clustering Interface":
    st.title("📥 Predict Cluster for New Input")
    st.write("Enter values to get cluster prediction:")
    brightness = st.number_input("Brightness", min_value=0.0, step=0.1)
    bright_t31 = st.number_input("Bright T31", min_value=0.0, step=0.1)
    frp = st.number_input("FRP", min_value=0.0, step=0.1)
    confidence = st.slider("Confidence", 0, 100)

    if st.button("Predict Cluster"):
        input_data = pd.DataFrame({
            "log_brightness": [np.log1p(brightness)],
            "log_bright_t31": [np.log1p(bright_t31)],
            "log_frp": [np.log1p(frp)],
            "confidence": [confidence]
        })
        scaler = StandardScaler()
        scaled_input = scaler.fit_transform(input_data)
        model = GaussianMixture(n_components=2, random_state=42).fit(data[["log_brightness", "log_bright_t31", "log_frp", "confidence"]])
        cluster_pred = model.predict(scaled_input)[0]
        st.success(f"Predicted GMM Cluster: {cluster_pred}")

# Comparative Analysis
elif page == "Comparative Analysis":
    st.title("📊 Comparative Clustering Analysis")
    cluster_cols = [col for col in data.columns if "cluster" in col]
    cluster_counts = pd.DataFrame({col: data[col].value_counts() for col in cluster_cols}).fillna(0).astype(int)
    st.dataframe(cluster_counts)

    fig, ax = plt.subplots(figsize=(10, 5))
    cluster_counts.plot(kind="bar", ax=ax)
    plt.title("Cluster Size Comparison Across Models")
    st.pyplot(fig)
