# SmartCart-Clustering-Sytem-
E-commerce Customer Segmentation System

An unsupervised machine learning system that segments SmartCart's e-commerce
customers into meaningful groups based on purchasing behaviour, engagement,
and loyalty — and turns those clusters into concrete marketing actions.

## Approach

The pipeline (developed in `SmartCart.ipynb`, and served via `app.py`)
follows these steps:

1. **Data Cleaning** — impute missing `Income` values with the median.
2. **Feature Engineering**
   - `Age` from `Year_Birth`
   - `Customer_Tenure_Days` from `Dt_Customer`
   - `Total_Spending` across all product categories
   - `Total_Children` from `Kidhome` + `Teenhome`
   - Simplified `Education` (Undergraduate / Graduate / Postgraduate)
   - Simplified `Living_With` (Alone / Partner) from `Marital_Status`
3. **Outlier Removal** — drop customers with `Age ≥ 90` or `Income ≥ 600,000`.
4. **Encoding & Scaling** — one-hot encode categorical features, then
   standardise all features with `StandardScaler`.
5. **Dimensionality Reduction** — PCA to 3 components for visualisation and
   clustering.
6. **Choosing k** — Elbow method (WCSS) and Silhouette score to pick the
   optimal number of clusters.
7. **Clustering** — KMeans and Agglomerative (Ward linkage) clustering on
   the PCA-reduced data.
8. **Cluster Profiling** — per-cluster feature means, visual comparisons,
   and rule-based persona labelling (💎 High-Value, 🌱 Growth/Standard,
   ⚠️ At-Risk/Low-Engagement) with suggested marketing actions.

## Repository Contents

| File | Description |
|---|---|
| `SmartCart.ipynb` | Original exploratory notebook: EDA, preprocessing, PCA, clustering |
| `app.py` | Interactive Streamlit app for exploring the full pipeline |
| `smartcart_customers.csv` | Source dataset (2,240 customers, 22 attributes) |
| `requirements.txt` | Python dependencies |

## Running the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`. From the sidebar
you can upload your own CSV (matching the same schema), switch between
KMeans and Agglomerative clustering, and adjust the number of clusters
live.

### App Tabs

- **📋 Data Overview** — row counts before/after cleaning, engineered features
- **🔍 EDA** — correlation heatmap, interactive scatter matrix
- **🧭 PCA** — variance explained, 3D customer projection
- **🎯 Choosing k** — elbow + silhouette charts with the suggested optimal k
- **🧩 Clusters** — 3D cluster visualisation, cluster sizes, income vs. spending
- **📊 Cluster Profiles** — feature means per cluster, downloadable labeled CSV
- **💡 Business Insights** — auto-generated personas and marketing
  recommendations per cluster

## Tech Stack

- **Python**, **pandas**, **NumPy**
- **scikit-learn** — `OneHotEncoder`, `StandardScaler`, `PCA`, `KMeans`,
  `AgglomerativeClustering`, `silhouette_score`
- **Streamlit** — interactive app front-end
- **Plotly** — interactive charts and 3D visualisations
