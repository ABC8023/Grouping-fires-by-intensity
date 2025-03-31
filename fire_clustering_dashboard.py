import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, fcluster
import hdbscan
from minisom import MiniSom
import skfuzzy as fuzz
from scipy.spatial.distance import cdist

st.set_page_config(page_title="Fire Intensity Clustering Dashboard", layout="wide")

st.title("🔥 Fire Intensity Clustering Dashboard")

# Sidebar
st.sidebar.header("Upload & Configuration")
uploaded_file = st.sidebar.file_uploader("Upload fire data CSV", type=["csv"])
model_option = st.sidebar.selectbox("Select Clustering Model", [
    "GMM", "Spectral", "DBSCAN", "HDBSCAN", "Fuzzy C-Means", "Hierarchical", "Adaptive Hierarchical", "SOM"
])

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
    df["log_frp"] = np.log1p(df["frp"])
    return df

def predict_nearest_cluster(input_scaled, model_labels, training_scaled_data):
    dists = cdist(input_scaled, training_scaled_data)
    nearest_index = np.argmin(dists)
    return model_labels[nearest_index]

if uploaded_file:
    data = load_and_clean_data(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.subheader("🔍 Data Exploration")

    with st.expander("🧮 Filter Data by Columns"):
        filtered_data = data.copy()
    
        # Define columns to allow filtering
        filter_columns = ["latitude", "longitude", "brightness", "confidence", "bright_t31", "frp", "daynight", "type"]
    
        for column in filter_columns:
            if pd.api.types.is_numeric_dtype(data[column]):
                min_val = float(data[column].min())
                max_val = float(data[column].max())
                selected_range = st.slider(
                    f"{column} range",
                    min_val, max_val,
                    (min_val, max_val),
                    key=f"{column}_filter"
                )
                filtered_data = filtered_data[filtered_data[column].between(*selected_range)]
            
            elif pd.api.types.is_object_dtype(data[column]) or pd.api.types.is_categorical_dtype(data[column]):
                options = data[column].dropna().unique().tolist()
                selected_options = st.multiselect(
                    f"Select {column}",
                    options,
                    default=options,
                    key=f"{column}_filter"
                )
                if selected_options:
                    filtered_data = filtered_data[filtered_data[column].isin(selected_options)]

    
    # Format acq_time and remove unwanted columns
    filtered_data_display = filtered_data.copy()
    
    # Convert acq_time like 320 → 03:20:00
    filtered_data_display["acq_time"] = pd.to_datetime(filtered_data_display["acq_time"].astype(str).str.zfill(4), format="%H%M", errors="coerce").dt.time
    
    # Drop log-transformed features from view
    filtered_data_display = filtered_data_display.drop(columns=["log_brightness", "log_frp"], errors="ignore")
    
    st.subheader("🔍 Filtered Data Preview")
    st.dataframe(filtered_data_display)


    # Feature Scaling
    features = ["log_brightness", "log_frp"]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[features])

    # PCA for visualization
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(reduced, columns=["PC1", "PC2"])

    st.subheader("📊 PCA Feature Importance (Explained Variance)")
    explained_var = pca.explained_variance_ratio_
    st.write(f"PC1 explains {explained_var[0]*100:.2f}% of variance.")
    st.write(f"PC2 explains {explained_var[1]*100:.2f}% of variance.")

    # Assign cluster labels to PCA for display
    pca_df["Cluster"] = labels
    
    # Plot after labels exist
    st.subheader(f"🌐 PCA Visualization - {model_option}")
    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color=pca_df["Cluster"].astype(str),
        title=f"PCA Clusters ({model_option})",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig, use_container_width=True)


    # Run selected model
    if model_option == "GMM":
        model = GaussianMixture(n_components=2, covariance_type="tied", init_params="kmeans", random_state=42)
        labels = model.fit_predict(scaled_data)

    elif model_option == "Spectral":
        model = SpectralClustering(n_clusters=2, affinity='rbf', assign_labels='kmeans', random_state=42)
        labels = model.fit_predict(scaled_data)

    elif model_option == "DBSCAN":
        model = DBSCAN(eps=1.0, min_samples=3)
        labels = model.fit_predict(scaled_data)

    elif model_option == "HDBSCAN":
        model = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=10)
        labels = model.fit_predict(scaled_data)

    elif model_option == "Fuzzy C-Means":
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            scaled_data.T, 2, m=1.5, error=0.005, maxiter=1000, init=None, seed=42
        )
        labels = np.argmax(u, axis=0)

    elif model_option == "Hierarchical":
        Z = linkage(scaled_data, method="ward")
        labels = fcluster(Z, t=3, criterion='maxclust')

    elif model_option == "Adaptive Hierarchical":
        Z = linkage(scaled_data, method="ward")
        labels = fcluster(Z, t=2, criterion='maxclust')

    elif model_option == "SOM":
        som = MiniSom(2, 2, len(features), sigma=0.5, learning_rate=0.5)
        som.random_weights_init(scaled_data)
        som.train_random(scaled_data, 200)
        winner_nodes = np.array([som.winner(x) for x in scaled_data])
        labels = np.array([r * 2 + c for r, c in winner_nodes])

    # Assign cluster labels to PCA for display
    pca_df["Cluster"] = labels

    st.subheader(f"🌐 PCA Visualization - {model_option}")
    fig = px.scatter(pca_df, x="PC1", y="PC2", color=pca_df["Cluster"].astype(str),
                     title=f"PCA Clusters ({model_option})", color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)

    # Show cluster size
    st.subheader("📊 Cluster Size Distribution")
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    st.bar_chart(cluster_counts)
    
    # Compute metrics if possible
    st.subheader("📊 Clustering Metrics")
    try:
        sil = silhouette_score(scaled_data, labels)
        dbi = davies_bouldin_score(scaled_data, labels)
        chi = calinski_harabasz_score(scaled_data, labels)
        st.metric("Silhouette Score", f"{sil:.3f}")
        st.metric("Davies-Bouldin Index", f"{dbi:.3f}")
        st.metric("Calinski-Harabasz Index", f"{chi:.0f}")
    except:
        st.warning("Not enough clusters to compute metrics.")

    # 🔢 Heatmap - All Numerical Features
    st.subheader("🔢 Feature Correlation Heatmap (All Features)")
    
    # Automatically detect numeric columns
    numeric_columns = data.select_dtypes(include=["int64", "float64"]).columns
    
    # Plot heatmap
    fig3, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data[numeric_columns].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig3)



    # Heatmap
    st.subheader("🔢 Feature Correlation Heatmap")
    fig3, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(data[["brightness", "bright_t31", "frp", "confidence"]].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig3)





    # 📈 Histogram - All Features
    st.subheader("📈 Interactive Feature Distribution")
    
    # Let user choose any numeric column
    col_to_plot = st.selectbox("Select Feature", numeric_columns)
    
    fig4, ax2 = plt.subplots()
    sns.histplot(data[col_to_plot], kde=True, ax=ax2, color="royalblue")
    ax2.set_title(f"Distribution of {col_to_plot}")
    st.pyplot(fig4)




    # Histogram
    st.subheader("📈 Interactive Feature Distribution")
    col_to_plot = st.selectbox("Select Feature", ["brightness", "bright_t31", "frp", "confidence"])
    fig4, ax2 = plt.subplots()
    sns.histplot(data[col_to_plot], kde=True, ax=ax2, color="royalblue")
    st.pyplot(fig4)




    # Comparative Analysis Section
    st.subheader("📊 Comparative Analysis Across Clustering Models")

    # Initialize dictionaries
    metric_results = {}
    models_to_compare = {
        "GMM": GaussianMixture(n_components=2, covariance_type="tied", init_params="kmeans", random_state=42),
        "Spectral": SpectralClustering(n_clusters=2, affinity='rbf', assign_labels='kmeans', random_state=42),
        "DBSCAN": DBSCAN(eps=1.0, min_samples=3),
        "HDBSCAN": hdbscan.HDBSCAN(min_cluster_size=10, min_samples=10),
        "Fuzzy C-Means": "fcm",
        "Hierarchical": "hier",
        "Adaptive Hierarchical": "adaptive",
        "SOM": "som"
    }

    for model_name, model in models_to_compare.items():
        try:
            if model_name == "Fuzzy C-Means":
                cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
                    scaled_data.T, 2, m=1.5, error=0.005, maxiter=1000, init=None, seed=42
                )
                labels = np.argmax(u, axis=0)
            elif model_name in ["Hierarchical", "Adaptive Hierarchical"]:
                Z = linkage(scaled_data, method="ward")
                labels = fcluster(Z, t=2, criterion='maxclust')
            elif model_name == "SOM":
                som = MiniSom(2, 2, len(features), sigma=0.5, learning_rate=0.5)
                som.random_weights_init(scaled_data)
                som.train_random(scaled_data, 200)
                winner_nodes = np.array([som.winner(x) for x in scaled_data])
                labels = np.array([r * 2 + c for r, c in winner_nodes])
            else:
                labels = model.fit_predict(scaled_data)

            if len(set(labels)) > 1:
                sil = silhouette_score(scaled_data, labels)
                dbi = davies_bouldin_score(scaled_data, labels)
                chi = calinski_harabasz_score(scaled_data, labels)
                metric_results[model_name] = {
                    "Silhouette": sil,
                    "DB Index": dbi,
                    "CH Index": chi
                }
        except:
            continue

    if metric_results:
        metric_df = pd.DataFrame(metric_results).T.round(3)
        st.dataframe(metric_df)

        # 📊 Comparative Analysis Across Clustering Models (Separate Charts)
        st.subheader("📊 Side-by-Side Metric Comparison")
        
        metric_results = {
            "Silhouette": {},        
            "Davies-Bouldin": {},
            "Calinski-Harabasz": {}
        }
        
        models_to_compare = {
            "GMM": GaussianMixture(n_components=2, covariance_type="tied", init_params="kmeans", random_state=42),
            "Spectral": SpectralClustering(n_clusters=2, affinity='rbf', assign_labels='kmeans', random_state=42),
            "DBSCAN": DBSCAN(eps=1.0, min_samples=3),
            "HDBSCAN": hdbscan.HDBSCAN(min_cluster_size=10, min_samples=10),
            "Fuzzy C-Means": "fcm",
            "Hierarchical": "hier",
            "Adaptive Hierarchical": "adapt",
            "SOM": "som"
        }
        
        for name, model in models_to_compare.items():
            try:
                if name == "Fuzzy C-Means":
                    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
                        scaled_data.T, 2, m=1.5, error=0.005, maxiter=1000, init=None, seed=42
                    )
                    labels = np.argmax(u, axis=0)
                elif name in ["Hierarchical", "Adaptive Hierarchical"]:
                    Z = linkage(scaled_data, method="ward")
                    labels = fcluster(Z, t=2, criterion='maxclust')
                elif name == "SOM":
                    som = MiniSom(2, 2, scaled_data.shape[1], sigma=0.5, learning_rate=0.5)
                    som.random_weights_init(scaled_data)
                    som.train_random(scaled_data, 200)
                    nodes = np.array([som.winner(x) for x in scaled_data])
                    labels = np.array([r * 2 + c for r, c in nodes])
                else:
                    labels = model.fit_predict(scaled_data)
        
                # Only compute if more than 1 cluster exists
                if len(set(labels)) > 1:
                    metric_results["Silhouette"][name] = silhouette_score(scaled_data, labels)
                    metric_results["Davies-Bouldin"][name] = davies_bouldin_score(scaled_data, labels)
                    metric_results["Calinski-Harabasz"][name] = calinski_harabasz_score(scaled_data, labels)
            except Exception as e:
                st.warning(f"Skipping {name} due to error: {e}")
        
        # Plot individual bar charts per metric
        import plotly.express as px
        
        for metric_name, results in metric_results.items():
            if results:
                df = pd.DataFrame({
                    "Model": list(results.keys()),
                    "Score": list(results.values())
                })
        
                fig = px.bar(df, x="Model", y="Score", title=f"{metric_name} Comparison",
                             color="Model", text_auto=True)
                st.plotly_chart(fig, use_container_width=True)

    # Predict interface (only for models that support it)
    st.subheader("🔢 Predict Cluster for New Observation")
    b = st.number_input("Brightness", value=330.0)
    frp = st.number_input("FRP", value=25.0)

    if st.button("Predict Cluster"):
        input_scaled = scaler.transform([[np.log1p(b), np.log1p(frp)]])
    
        if model_option == "Fuzzy C-Means":
            cluster_pred = model.predict(input_scaled)[0]
        else:
            cluster_pred = predict_nearest_cluster(input_scaled, labels, scaled_data)
    
        st.success(f"Predicted Cluster: {cluster_pred}")

        # Optional: Add cluster interpretation if you have it
        cluster_desc = {
            0: "Low intensity fire zone",
            1: "High intensity fire zone"
        }
        st.info(cluster_desc.get(cluster_pred, "Unknown Cluster"))

else:
    st.info("Please upload a fire dataset CSV to get started.")
