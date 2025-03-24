import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

st.set_page_config(page_title="Fire Intensity Clustering Dashboard", layout="wide")

st.title("🔥 Fire Intensity Clustering Dashboard")

# Sidebar
st.sidebar.header("Upload & Configuration")
uploaded_file = st.sidebar.file_uploader("Upload fire data CSV", type=["csv"])
model_option = st.sidebar.selectbox("Select Clustering Model", ["GMM"])  # Can expand later with more models

# Load and preprocess data
@st.cache_data
def load_and_clean_data(csv):
    df = pd.read_csv(csv)
    df = df[df["type"] != 3]  # remove offshore
    df = df[
        (df["latitude"].between(-90, 90)) &
        (df["longitude"].between(-180, 180)) &
        (df["brightness"] > 200) &
        (df["bright_t31"] > 200) &
        (df["frp"] >= 0) &
        (df["confidence"].between(0, 100))
    ]
    df.drop_duplicates(inplace=True)
    df["log_brightness"] = np.log1p(df["brightness"])
    df["log_bright_t31"] = np.log1p(df["bright_t31"])
    df["log_frp"] = np.log1p(df["frp"])
    return df

if uploaded_file:
    data = load_and_clean_data(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(data)

    # Feature Scaling
    features = ["log_brightness", "log_bright_t31", "log_frp", "confidence"]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[features])

    # PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(reduced, columns=["PC1", "PC2"])

    st.subheader("🌐 PCA Visualization")
    fig = px.scatter(pca_df, x="PC1", y="PC2", title="PCA Projection (Unlabeled)")
    st.plotly_chart(fig, use_container_width=True)

    # Apply GMM
    gmm = GaussianMixture(n_components=2, covariance_type="tied", init_params="kmeans", random_state=42)
    labels = gmm.fit_predict(scaled_data)
    pca_df["Cluster"] = labels

    st.subheader("🎯 GMM Clustering Results")
    fig2 = px.scatter(pca_df, x="PC1", y="PC2", color=pca_df["Cluster"].astype(str),
                      title="GMM Clusters", color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig2, use_container_width=True)

    # Evaluation metrics
    silhouette = silhouette_score(scaled_data, labels)
    db_index = davies_bouldin_score(scaled_data, labels)
    ch_index = calinski_harabasz_score(scaled_data, labels)

    st.subheader("📊 Clustering Metrics")
    metrics_df = pd.DataFrame({
        "Model": ["GMM"],
        "Silhouette Score": [silhouette],
        "Davies-Bouldin Index": [db_index],
        "Calinski-Harabasz Index": [ch_index]
    })
    st.dataframe(metrics_df)

    # Feature importance heatmap
    st.subheader("🔢 Feature Correlation Heatmap")
    corr = data[["brightness", "bright_t31", "frp", "confidence"]].corr()
    fig3, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig3)

    # Feature exploration
    st.subheader("📈 Interactive Feature Distribution")
    col_to_plot = st.selectbox("Select Feature", ["brightness", "bright_t31", "frp", "confidence"], key="feature_select")
    fig4, ax2 = plt.subplots()
    sns.histplot(data[col_to_plot], kde=True, ax=ax2, color="royalblue")
    st.pyplot(fig4)

    # Optional: Predict cluster for user input
    st.subheader("🔢 Predict Cluster for New Fire Observation")
    b = st.number_input("Brightness", value=330.0)
    b31 = st.number_input("Brightness_T31", value=310.0)
    frp = st.number_input("FRP", value=25.0)
    conf = st.slider("Confidence", 0, 100, 80)

    if st.button("Predict Cluster"):
        input_scaled = scaler.transform([[np.log1p(b), np.log1p(b31), np.log1p(frp), conf]])
        cluster_pred = gmm.predict(input_scaled)[0]
        st.success(f"Predicted Cluster: {cluster_pred}")

    # Comparative bar chart (more models can be added)
    st.subheader("📊 Comparative Clustering Metrics")
    fig_comp = px.bar(
        metrics_df.melt(id_vars=["Model"], var_name="Metric", value_name="Score"),
        x="Metric", y="Score", color="Model", barmode="group",
        title="Comparison of Clustering Metrics"
    )
    st.plotly_chart(fig_comp, use_container_width=True)

else:
    st.info("Please upload a fire dataset CSV to get started.")
