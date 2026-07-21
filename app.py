"""
SmartCart Customer Segmentation
--------------------------------
Interactive Streamlit app that mirrors the SmartCart.ipynb pipeline:
data cleaning -> feature engineering -> encoding/scaling -> PCA ->
k selection (elbow + silhouette) -> clustering -> cluster profiling.
"""

import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

st.set_page_config(page_title="SmartCart Segmentation", page_icon="🛒", layout="wide")

# --------------------------------------------------------------------------
# Data loading & pipeline (cached so it only reruns when inputs change)
# --------------------------------------------------------------------------


@st.cache_data
def load_data(file) -> pd.DataFrame:
    return pd.read_csv(file)


@st.cache_data
def preprocess(df_raw: pd.DataFrame):
    df = df_raw.copy()

    # 1. Missing values
    df["Income"] = df["Income"].fillna(df["Income"].median())

    # 2. Feature engineering
    df["Age"] = 2026 - df["Year_Birth"]
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
    reference_date = df["Dt_Customer"].max()
    df["Customer_Tenure_Days"] = (reference_date - df["Dt_Customer"]).dt.days
    df["Total_Spending"] = (
        df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"]
        + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
    )
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

    df["Education"] = df["Education"].replace({
        "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
        "Graduation": "Graduate",
        "Master": "Postgraduate", "PhD": "Postgraduate",
    })
    df["Living_With"] = df["Marital_Status"].replace({
        "Married": "Partner", "Together": "Partner",
        "Single": "Alone", "Divorced": "Alone",
        "Widow": "Alone", "Absurd": "Alone", "YOLO": "Alone",
    })

    # 3. Drop redundant columns
    cols = ["ID", "Year_Birth", "Marital_Status", "Kidhome", "Teenhome", "Dt_Customer"]
    spending_cols = ["MntWines", "MntFruits", "MntMeatProducts",
                      "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    df_cleaned = df.drop(columns=cols + spending_cols)

    # 4. Outlier removal
    n_before = len(df_cleaned)
    df_cleaned = df_cleaned[(df_cleaned["Age"] < 90) & (df_cleaned["Income"] < 600_000)]
    n_after = len(df_cleaned)

    return df_cleaned.reset_index(drop=True), n_before, n_after


@st.cache_data
def encode_scale(df_cleaned: pd.DataFrame):
    cat_cols = ["Education", "Living_With"]
    ohe = OneHotEncoder()
    enc_cols = ohe.fit_transform(df_cleaned[cat_cols])
    enc_df = pd.DataFrame(
        enc_cols.toarray(),
        columns=ohe.get_feature_names_out(cat_cols),
        index=df_cleaned.index,
    )
    X = pd.concat([df_cleaned.drop(columns=cat_cols), enc_df], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled


@st.cache_data
def run_pca(X_scaled: np.ndarray, n_components: int = 3):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca.explained_variance_ratio_


@st.cache_data
def elbow_silhouette(X_pca: np.ndarray, k_max: int = 10):
    wcss = []
    for k in range(1, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_pca)
        wcss.append(km.inertia_)

    sil_scores = []
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_pca)
        sil_scores.append(silhouette_score(X_pca, labels))

    return wcss, sil_scores


def cluster(X_pca: np.ndarray, algo: str, k: int):
    if algo == "KMeans":
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
    else:
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = model.fit_predict(X_pca)
    return labels


PALETTE = px.colors.qualitative.Set1


def build_personas(cluster_summary: pd.DataFrame) -> pd.DataFrame:
    """Rank clusters relative to each other and assign a business persona +
    a suggested marketing action. Works for any k since it's relative, not
    based on fixed thresholds."""

    def z(col):
        vals = cluster_summary[col]
        std = vals.std(ddof=0)
        return (vals - vals.mean()) / std if std > 0 else vals * 0

    score = pd.Series(0.0, index=cluster_summary.index)
    if "Income" in cluster_summary:
        score += z("Income")
    if "Total_Spending" in cluster_summary:
        score += z("Total_Spending")
    if "Recency" in cluster_summary:
        score -= z("Recency")          # higher recency = longer since last purchase = bad
    if "Complain" in cluster_summary:
        score -= z("Complain")
    if "Response" in cluster_summary:
        score += 0.5 * z("Response")   # responsive to past campaigns = engaged

    ranked = score.rank(ascending=False, method="first")
    n = len(ranked)

    personas, actions = [], []
    for cl in cluster_summary.index:
        r = ranked[cl]
        if r <= max(1, round(n / 3)):
            personas.append("💎 High-Value")
            actions.append(
                "Prioritise retention: early access, loyalty perks, premium bundles. "
                "Low discounting needed — protect margin, not price-chase."
            )
        elif r > n - max(1, round(n / 3)):
            personas.append("⚠️ At-Risk / Low-Engagement")
            actions.append(
                "Win-back campaigns: reactivation discounts, reminder emails, "
                "surveys to diagnose disengagement before they churn."
            )
        else:
            personas.append("🌱 Growth / Standard")
            actions.append(
                "Upsell & cross-sell: personalised recommendations, targeted "
                "promotions to nudge them toward the High-Value segment."
            )
    result = cluster_summary.copy()
    result.insert(0, "Persona", personas)
    result.insert(1, "Suggested Action", actions)
    return result

# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------

st.sidebar.title("🛒 SmartCart Controls")

uploaded = st.sidebar.file_uploader("Upload customer CSV", type="csv")
if uploaded is not None:
    df_raw = load_data(uploaded)
    st.sidebar.success(f"Loaded {uploaded.name}")
else:
    df_raw = load_data("smartcart_customers.csv")
    st.sidebar.info("Using bundled smartcart_customers.csv")

algo = st.sidebar.radio("Clustering algorithm", ["KMeans", "Agglomerative"])
k_max = st.sidebar.slider("Max k to test (elbow/silhouette)", 4, 15, 10)

df_cleaned, n_before, n_after = preprocess(df_raw)
X, X_scaled = encode_scale(df_cleaned)
X_pca, explained_var = run_pca(X_scaled, n_components=3)
wcss, sil_scores = elbow_silhouette(X_pca, k_max=k_max)

best_k_by_silhouette = int(np.argmax(sil_scores)) + 2  # scores start at k=2

k = st.sidebar.slider(
    "Number of clusters (k)", 2, k_max,
    value=min(4, k_max),
    help=f"Silhouette score suggests k = {best_k_by_silhouette}",
)
st.sidebar.caption(f"💡 Silhouette-suggested k: **{best_k_by_silhouette}**")

labels = cluster(X_pca, algo, k)
X_labeled = X.copy()
X_labeled["Cluster"] = labels.astype(str)

# --------------------------------------------------------------------------
# Main layout
# --------------------------------------------------------------------------

st.title("🛒 SmartCart Customer Segmentation")
st.caption("Explore how customers cluster based on spending, income, and demographics.")

tab_overview, tab_eda, tab_pca, tab_k, tab_clusters, tab_profiles, tab_insights = st.tabs(
    ["📋 Data Overview", "🔍 EDA", "🧭 PCA", "🎯 Choosing k", "🧩 Clusters",
     "📊 Cluster Profiles", "💡 Business Insights"]
)

# ---- Data Overview --------------------------------------------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw rows", n_before)
    c2.metric("After cleaning", n_after)
    c3.metric("Outliers removed", n_before - n_after)
    c4.metric("Features used", X.shape[1])

    st.subheader("Cleaned & engineered data")
    st.dataframe(df_cleaned.head(20), use_container_width=True)

    with st.expander("Show raw uploaded data"):
        st.dataframe(df_raw.head(20), use_container_width=True)

# ---- EDA -------------------------------------------------------------------
with tab_eda:
    st.subheader("Correlation heatmap")
    corr = df_cleaned.corr(numeric_only=True)
    fig_corr = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        aspect="auto",
    )
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Feature relationships")
    default_cols = ["Income", "Recency", "Response", "Age", "Total_Spending", "Total_Children"]
    default_cols = [c for c in default_cols if c in df_cleaned.columns]
    pair_cols = st.multiselect(
        "Columns to compare", options=list(df_cleaned.select_dtypes(include=np.number).columns),
        default=default_cols,
    )
    if len(pair_cols) >= 2:
        fig_pair = px.scatter_matrix(df_cleaned, dimensions=pair_cols, height=700)
        fig_pair.update_traces(diagonal_visible=False, showupperhalf=False, marker=dict(size=3))
        st.plotly_chart(fig_pair, use_container_width=True)
    else:
        st.info("Select at least two columns to see a scatter matrix.")

# ---- PCA ---------------------------------------------------------------
with tab_pca:
    st.subheader("Variance explained by principal components")
    c1, c2, c3 = st.columns(3)
    for col, i in zip([c1, c2, c3], range(3)):
        col.metric(f"PC{i+1}", f"{explained_var[i]*100:.1f}%")
    st.caption(f"Total variance captured by 3 components: {explained_var.sum()*100:.1f}%")

    st.subheader("3D projection of customers")
    fig_pca = go.Figure(data=[go.Scatter3d(
        x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
        mode="markers", marker=dict(size=3, opacity=0.7),
    )])
    fig_pca.update_layout(
        scene=dict(xaxis_title="PCA1", yaxis_title="PCA2", zaxis_title="PCA3"),
        height=650, margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_pca, use_container_width=True)

# ---- Choosing k ----------------------------------------------------------
with tab_k:
    st.subheader("Elbow method (WCSS) & Silhouette score")
    ks = list(range(1, k_max + 1))
    ks_sil = list(range(2, k_max + 1))

    fig_k = go.Figure()
    fig_k.add_trace(go.Scatter(x=ks, y=wcss, mode="lines+markers", name="WCSS", yaxis="y1"))
    fig_k.add_trace(go.Scatter(x=ks_sil, y=sil_scores, mode="lines+markers",
                                name="Silhouette", yaxis="y2", line=dict(dash="dash")))
    fig_k.update_layout(
        xaxis=dict(title="k"),
        yaxis=dict(title="WCSS", side="left"),
        yaxis2=dict(title="Silhouette score", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.1),
        height=500,
    )
    fig_k.add_vline(x=k, line_dash="dot", line_color="green",
                     annotation_text=f"selected k={k}", annotation_position="top")
    st.plotly_chart(fig_k, use_container_width=True)
    st.info(f"The silhouette score peaks at **k = {best_k_by_silhouette}**. "
            f"You currently have k = **{k}** selected in the sidebar.")

# ---- Clusters ------------------------------------------------------------
with tab_clusters:
    st.subheader(f"{algo} clustering with k = {k}")

    fig_clusters = go.Figure()
    for i, cl in enumerate(sorted(X_labeled["Cluster"].unique(), key=int)):
        mask = X_labeled["Cluster"] == cl
        fig_clusters.add_trace(go.Scatter3d(
            x=X_pca[mask.values, 0], y=X_pca[mask.values, 1], z=X_pca[mask.values, 2],
            mode="markers", name=f"Cluster {cl}",
            marker=dict(size=3, color=PALETTE[i % len(PALETTE)], opacity=0.75),
        ))
    fig_clusters.update_layout(
        scene=dict(xaxis_title="PCA1", yaxis_title="PCA2", zaxis_title="PCA3"),
        height=650, margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_clusters, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Cluster sizes")
        counts = X_labeled["Cluster"].value_counts().sort_index()
        fig_counts = px.bar(
            x=counts.index, y=counts.values, color=counts.index,
            color_discrete_sequence=PALETTE,
            labels={"x": "Cluster", "y": "Number of customers"},
        )
        st.plotly_chart(fig_counts, use_container_width=True)

    with c2:
        st.subheader("Income vs. Spending")
        fig_scatter = px.scatter(
            X_labeled, x="Total_Spending", y="Income", color="Cluster",
            color_discrete_sequence=PALETTE, opacity=0.7,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# ---- Cluster profiles -----------------------------------------------------
with tab_profiles:
    st.subheader("Cluster summary (feature means)")
    numeric_labeled = X_labeled.copy()
    numeric_labeled["Cluster"] = numeric_labeled["Cluster"].astype(int)
    cluster_summary = numeric_labeled.groupby("Cluster").mean(numeric_only=True)
    st.dataframe(cluster_summary.style.format("{:.2f}"), use_container_width=True)

    st.subheader("Compare a feature across clusters")
    profile_feature = st.selectbox(
        "Feature", options=[c for c in cluster_summary.columns], index=0
    )
    fig_profile = px.bar(
        x=cluster_summary.index.astype(str), y=cluster_summary[profile_feature],
        color=cluster_summary.index.astype(str), color_discrete_sequence=PALETTE,
        labels={"x": "Cluster", "y": profile_feature},
    )
    st.plotly_chart(fig_profile, use_container_width=True)

    csv_export = X_labeled.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download labeled customer data (CSV)",
        data=csv_export,
        file_name="smartcart_customers_clustered.csv",
        mime="text/csv",
    )

# ---- Business Insights -----------------------------------------------------
with tab_insights:
    st.subheader("From clusters to marketing action")
    st.caption(
        "SmartCart's problem statement asks for more than clusters — it asks for "
        "personalised marketing and retention decisions. This tab translates each "
        "cluster's stats into a persona and a concrete next step."
    )

    persona_table = build_personas(cluster_summary)

    # Headline cards
    n_high = (persona_table["Persona"] == "💎 High-Value").sum()
    n_risk = (persona_table["Persona"] == "⚠️ At-Risk / Low-Engagement").sum()
    n_growth = (persona_table["Persona"] == "🌱 Growth / Standard").sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("💎 High-Value clusters", n_high)
    c2.metric("🌱 Growth clusters", n_growth)
    c3.metric("⚠️ At-Risk clusters", n_risk)

    st.subheader("Persona summary")
    display_cols = [c for c in ["Persona", "Suggested Action", "Income", "Total_Spending",
                                 "Recency", "Complain", "Response", "Customer_Tenure_Days"]
                     if c in persona_table.columns]
    st.dataframe(
        persona_table[display_cols].style.format(
            {c: "{:.2f}" for c in display_cols if c not in ("Persona", "Suggested Action")}
        ),
        use_container_width=True,
    )

    st.subheader("Customer count per persona")
    persona_counts = numeric_labeled["Cluster"].map(
        persona_table["Persona"].to_dict()
    ).value_counts()
    fig_persona = px.pie(
        names=persona_counts.index, values=persona_counts.values,
        color=persona_counts.index,
        color_discrete_map={
            "💎 High-Value": "#2ca02c",
            "🌱 Growth / Standard": "#1f77b4",
            "⚠️ At-Risk / Low-Engagement": "#d62728",
        },
        hole=0.4,
    )
    st.plotly_chart(fig_persona, use_container_width=True)

    st.subheader("Per-cluster recommendation")
    for cl in cluster_summary.index:
        with st.expander(f"Cluster {cl} — {persona_table.loc[cl, 'Persona']}"):
            st.write(persona_table.loc[cl, "Suggested Action"])
            st.write(
                f"Avg income: **{cluster_summary.loc[cl, 'Income']:.0f}** · "
                f"Avg spending: **{cluster_summary.loc[cl, 'Total_Spending']:.0f}** · "
                f"Avg recency (days since last purchase): **{cluster_summary.loc[cl, 'Recency']:.0f}**"
                if "Recency" in cluster_summary.columns and "Income" in cluster_summary.columns
                else "Stats unavailable for this cluster."
            )

    st.caption(
        "Note: personas are assigned *relative to the other clusters found in this run* — "
        "they are not fixed thresholds, so they'll adapt automatically if you change k, "
        "the algorithm, or the uploaded dataset."
    )