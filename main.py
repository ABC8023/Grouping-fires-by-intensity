import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
import plotly.graph_objects as go
import datetime
from skfuzzy import cmeans_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

st.set_page_config(page_title="🔥 Fire Intensity - Fuzzy C-Means Only", layout="wide")

# Navigation
page = st.sidebar.selectbox("Select a page", ["Dataset Dashboard", "Cluster Prediction"])

if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []



if page == "Dataset Dashboard":
    st.title("🔥 Fire Intensity Clustering - Dataset Dashboard")

    st.sidebar.header("Upload Your Data")
    uploaded_file = st.sidebar.file_uploader("Upload fire data CSV", type=["csv"])

    @st.cache_data
    def load_and_clean_data(csv):
        df = pd.read_csv(csv)
        df = df[df["type"] != 3]
        df = df[
            (df["latitude"].between(-90, 90)) &
            (df["longitude"].between(-180, 180)) &
            (df["brightness"] > 200) &
            (df["bright_t31"] > 200) &
            (df["frp"] >= 0) &
            (df["confidence"].between(0, 100))
        ]
        df = df[df.isnull().sum(axis=1) <= 5]
        df.drop_duplicates(inplace=True)
        df["log_brightness"] = np.log1p(df["brightness"])
        df["log_frp"] = np.log1p(df["frp"])
        return df

    if uploaded_file:
        data = load_and_clean_data(uploaded_file)
        st.subheader("📄 Dataset Preview")
        st.subheader("🔍 Data Exploration")

        with st.expander("🧮 Filter Data by Columns"):
            filtered_data = data.copy()
            filter_columns = ["latitude", "longitude", "brightness", "confidence", "bright_t31", "frp", "daynight", "type"]
            for column in filter_columns:
                if pd.api.types.is_numeric_dtype(data[column]):
                    min_val = float(data[column].min())
                    max_val = float(data[column].max())
                    selected_range = st.slider(f"{column} range", min_val, max_val, (min_val, max_val), key=f"{column}_filter")
                    filtered_data = filtered_data[filtered_data[column].between(*selected_range)]
                elif pd.api.types.is_object_dtype(data[column]) or pd.api.types.is_categorical_dtype(data[column]):
                    options = data[column].dropna().unique().tolist()
                    selected_options = st.multiselect(f"Select {column}", options, default=options, key=f"{column}_filter")
                    if selected_options:
                        filtered_data = filtered_data[filtered_data[column].isin(selected_options)]

        filtered_data_display = filtered_data.copy()
        filtered_data_display["acq_time"] = pd.to_datetime(
            filtered_data_display["acq_time"].astype(str).str.zfill(4),
            format="%H%M", errors="coerce").dt.time
        filtered_data_display = filtered_data_display.drop(columns=["log_brightness", "log_frp"], errors="ignore")
        st.subheader("🔍 Filtered Data Preview")
        st.dataframe(filtered_data_display)

        features = ["log_brightness", "log_frp"]
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[features])

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(scaled_data)
        pca_df = pd.DataFrame(reduced, columns=["PC1", "PC2"])

        st.subheader("📊 PCA Feature Importance (Explained Variance)")
        explained_var = pca.explained_variance_ratio_
        st.write(f"PC1 explains {explained_var[0]*100:.2f}% of variance.")
        st.write(f"PC2 explains {explained_var[1]*100:.2f}% of variance.")

        model_bundle = joblib.load("fcm_model_bundle.pkl")
        cntr = model_bundle["cntr"]
        best_m = model_bundle["m"]
        scaler = model_bundle["scaler"]
        error = model_bundle["error"]
        maxiter = model_bundle["maxiter"]


        u, *_ = cmeans_predict(scaled_data.T, cntr, best_m, error, maxiter)

        labels = np.argmax(u, axis=0)

        pca_df["Cluster"] = labels
        fig = px.scatter(
            pca_df, x="PC1", y="PC2", color=pca_df["Cluster"].astype(str),
            title="PCA Clusters (Fuzzy C-Means)", color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Cluster Size Distribution")
        st.bar_chart(pd.Series(labels).value_counts().sort_index())

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

        st.subheader("🔢 Feature Correlation Heatmap (All Features)")

        # Explicitly select relevant features for the heatmap
        heatmap_features = [
            "latitude", "longitude", "brightness", "scan", "track",
            "confidence", "bright_t31", "frp", "type"
        ]

        # Keep only the columns that exist in the current dataset
        heatmap_features = [col for col in heatmap_features if col in data.columns]

        corr = data[heatmap_features].corr()

        fig3, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        ax.set_title("Correlation Heatmap of All Numerical Features")
        st.pyplot(fig3)

        st.subheader("📈 Interactive Feature Distribution")
        numeric_columns = data.select_dtypes(include=["int64", "float64"]).columns
        col_to_plot = st.selectbox("Select Feature", numeric_columns)
        fig4, ax2 = plt.subplots()
        sns.histplot(data[col_to_plot], kde=True, ax=ax2, color="royalblue")
        ax2.set_title(f"Distribution of {col_to_plot}")
        st.pyplot(fig4)

    else:
        st.info("Please upload a fire dataset CSV to get started.")

elif page == "Cluster Prediction":
    
    st.title("🔮 Predict Fire Cluster")
    st.subheader("📥 Input Brightness and FRP")
    brightness = st.number_input("Brightness", min_value=0.0, value=330.0)
    frp = st.number_input("FRP", min_value=0.0, value=25.0)

    if st.button("Predict Cluster"):
        model_bundle = joblib.load("fcm_model_bundle.pkl")
        cntr = model_bundle["cntr"]
        best_m = model_bundle["m"]
        scaler = model_bundle["scaler"]
        error = model_bundle["error"]
        maxiter = model_bundle["maxiter"]
        input_scaled = scaler.transform([[np.log1p(brightness), np.log1p(frp)]])
        u_pred, *_ = cmeans_predict(input_scaled.T, cntr, best_m, error, maxiter)
        cluster_pred = np.argmax(u_pred)
        membership_probs = u_pred[:, 0]

        st.success(f"Predicted Cluster: {cluster_pred}")
        
        # Cluster interpretation
        cluster_desc = {
            0: "🔥 Cluster 0: Low-intensity fire zone — typically controlled burns or early-stage spread.",
            1: "🔥 Cluster 1: High-intensity fire zone — requires immediate attention and may be dangerous.",
        }
        st.info(cluster_desc.get(cluster_pred, "⚠ Unknown cluster type."))
        
        # Low confidence warning
        if max(membership_probs) < 0.6:
            st.warning("⚠ Warning: Low confidence in cluster assignment. Fire characteristics are ambiguous.")


        # 📊 Membership bar chart
        st.subheader("📊 Cluster Membership Confidence")
        bar_fig = go.Figure(data=[
            go.Bar(x=[f"Cluster {i}" for i in range(len(membership_probs))],
                   y=membership_probs,
                   marker_color="orange")
        ])
        bar_fig.update_layout(
            yaxis=dict(title="Membership Strength", range=[0, 1]),
            xaxis=dict(title="Clusters"),
            title="Fuzzy Membership Strength per Cluster"
        )
        st.plotly_chart(bar_fig, use_container_width=True)

        # Record prediction with timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["prediction_history"].append({
            "Timestamp": timestamp,
            "Brightness": brightness,
            "FRP": frp,
            "Predicted Cluster": cluster_pred,
            "Membership Probs": membership_probs.tolist()
        })

    # Show prediction history and download
    if st.session_state["prediction_history"]:
        st.subheader("📜 Prediction History")
        history_df = pd.DataFrame(st.session_state["prediction_history"])
        st.dataframe(history_df)

        csv_history = history_df.to_csv(index=False)
        st.download_button("📥 Download Full Prediction History", csv_history, "prediction_history.csv", "text/csv")

