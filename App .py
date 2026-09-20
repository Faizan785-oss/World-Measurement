#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


# In[2]:


st.set_page_config(page_title="World Development Clustering", layout="wide")

st.title("World Development Measurement – Clustering")
st.write("Upload the Excel dataset to perform clustering and explore country groups.")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, sheet_name="world_development")

    text_numeric_cols = [
        "Business Tax Rate",
        "GDP",
        "Health Exp/Capita",
        "Tourism Inbound",
        "Tourism Outbound"
    ]

    for col in text_numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r"[$,%]", "", regex=True),
                errors="coerce"
            )

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    features = [
        c for c in df.columns
        if c not in ["Country", "Number of Records"]
    ]

    X = df[features].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    st.sidebar.header("Clustering Settings")
    k = st.sidebar.slider("Number of clusters", min_value=2, max_value=10, value=3)

    model_choice = st.sidebar.selectbox(
        "Model",
        ["K-Means", "Agglomerative", "GMM"]
    )

    if model_choice == "K-Means":
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
    elif model_choice == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    else:
        model = GaussianMixture(n_components=k, random_state=42, n_init=10)

    labels = model.fit_predict(X_scaled)
    df["Cluster"] = labels

    st.subheader("Cluster Metrics")

    metrics = {
        "Silhouette Score": silhouette_score(X_scaled, labels),
        "Davies-Bouldin Index": davies_bouldin_score(X_scaled, labels),
        "Calinski-Harabasz Score": calinski_harabasz_score(X_scaled, labels)
    }

    c1, c2, c3 = st.columns(3)
    c1.metric("Silhouette", f"{metrics['Silhouette Score']:.4f}")
    c2.metric("Davies-Bouldin", f"{metrics['Davies-Bouldin Index']:.4f}")
    c3.metric("Calinski-Harabasz", f"{metrics['Calinski-Harabasz Score']:.2f}")

    st.subheader("Cluster Counts")
    st.dataframe(
        df["Cluster"].value_counts().sort_index().rename("Count").to_frame(),
        use_container_width=True
    )

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    st.subheader("PCA Cluster Visualization")
    fig = plt.figure(figsize=(9, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=20)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"PCA – {model_choice}")
    st.pyplot(fig)

    st.subheader("Clustered Dataset")
    st.dataframe(df, use_container_width=True)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Clustered CSV",
        data=csv_data,
        file_name="world_development_clustered.csv",
        mime="text/csv"
    )

else:
    st.info("Upload P706_ World_development_mesurement.xlsx to start.")


# In[ ]:




