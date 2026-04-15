import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, LabelEncoder, OrdinalEncoder
from sklearn.ensemble import (
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score,
    r2_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    roc_curve,
)


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="House Price Prediction Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -------------------------------------------------
# THEME / CSS
# -------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg: #081120;
    --panel: #0d1b2a;
    --panel-2: #13253a;
    --panel-3: #1b314d;
    --border: #27496d;
    --accent: #4da3ff;
    --accent-2: #7cc2ff;
    --text: #ecf5ff;
    --muted: #97acc4;
    --success: #6ed8a5;
    --warning: #ffc857;
    --danger: #ff7b7b;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
    background: linear-gradient(180deg, #07101d 0%, #081120 100%) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

h1, h2, h3, h4 {
    font-family: 'Playfair Display', serif !important;
    color: var(--text) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}

.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.02) !important;
    border-radius: 12px 12px 0 0 !important;
    padding: 12px 18px !important;
    color: var(--muted) !important;
}

.stTabs [aria-selected="true"] {
    color: white !important;
    background: linear-gradient(135deg, #2b67c9, #4da3ff) !important;
}

div[data-testid="stButton"] > button {
    border: none !important;
    border-radius: 12px !important;
    background: linear-gradient(135deg, #2b67c9, #4da3ff) !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.65rem 1rem !important;
}

[data-testid="stFileUploader"] {
    border: 1px dashed var(--border) !important;
    border-radius: 16px !important;
    background: rgba(255,255,255,0.02) !important;
    padding: 16px !important;
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    padding: 14px;
    border-radius: 16px;
}

.hero {
    text-align: center;
    padding: 28px 0 10px 0;
}

.hero-icon {
    font-size: 54px;
    margin-bottom: 6px;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(32px, 5vw, 58px);
    font-weight: 700;
    color: white;
}

.hero-sub {
    color: var(--muted);
    font-size: 14px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 8px;
}

.pipeline-rail {
    display: flex;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    margin: 20px 0 28px 0;
}

.rail-step {
    flex: 1;
    padding: 12px 6px;
    text-align: center;
    font-size: 11px;
    color: var(--muted);
    background: rgba(255,255,255,0.02);
    border-right: 1px solid rgba(255,255,255,0.06);
}

.rail-step:last-child {
    border-right: none;
}

.rail-step.done {
    background: rgba(77,163,255,0.10);
    color: #dcecff;
}

.rail-step.active {
    background: linear-gradient(135deg, #2b67c9, #4da3ff);
    color: white;
    font-weight: 700;
}

.section-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 16px;
}

.badge-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 18px;
}

.badge-num {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #2b67c9, #4da3ff);
    color: white;
    font-weight: 700;
}

.badge-title {
    font-size: 28px;
    font-family: 'Playfair Display', serif;
    color: white;
}

.small-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    background: rgba(77,163,255,0.12);
    border: 1px solid rgba(77,163,255,0.22);
    color: #dcecff;
    font-size: 12px;
    margin-right: 8px;
}
</style>
""",
    unsafe_allow_html=True,
)


# -------------------------------------------------
# SESSION STATE DEFAULTS
# -------------------------------------------------
defaults = {
    "df": None,
    "problem_type": "Regression",
    "target_col": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "best_model": None,
    "selected_features": None,
    "pipeline_stage": 0,
    "model_comparison": {},
    "_scaler": None,
    "_Xte_scaled": None,
    "_yte": None,
    "_Xtr_scaled": None,
    "_ytr": None,
    "tuned_model": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(family="Inter, sans-serif", color="#dbeafe", size=12),
    xaxis=dict(gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.08)"),
)

BLUE = "#4da3ff"
LIGHT_BLUE = "#7cc2ff"
TEAL = "#57d3c0"
ORANGE = "#ffb454"
RED = "#ff7b7b"
PURPLE = "#ae8bff"
PALETTE = [BLUE, TEAL, ORANGE, PURPLE, LIGHT_BLUE]
COLOR_SCALE = [[0, "#0b1a2d"], [0.5, "#2b67c9"], [1, "#9bd1ff"]]


def plot_theme(**kwargs):
    layout = THEME.copy()
    layout.update(kwargs)
    return layout


def step_badge(number, title):
    st.markdown(
        f"""
        <div class="badge-row">
            <div class="badge-num">{number}</div>
            <div class="badge-title">{title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def get_download_link(obj, filename, label):
    buffer = pickle.dumps(obj)
    b64 = base64.b64encode(buffer).decode()
    return (
        f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}" '
        f'style="display:inline-block;padding:10px 16px;border-radius:12px;text-decoration:none;'
        f'background:linear-gradient(135deg,#2b67c9,#4da3ff);color:white;font-weight:600;">{label}</a>'
    )


def infer_default_target(columns):
    preferred = [
        "price",
        "median_house_value",
        "SalePrice",
        "sale_price",
        "house_price",
        "target",
    ]
    for col in preferred:
        if col in columns:
            return col
    return columns[-1] if len(columns) else None


# -------------------------------------------------
# MODEL REGISTRY
# -------------------------------------------------
REGRESSION_MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01),
    "SVR (RBF)": SVR(kernel="rbf"),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
}

CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "SVC (RBF)": SVC(kernel="rbf", probability=True),
    "KNN Classifier": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

GRIDS = {
    "LinearRegression": {"fit_intercept": [True, False]},
    "Ridge": {"alpha": [0.001, 0.01, 0.1, 1, 10, 100]},
    "Lasso": {"alpha": [0.001, 0.01, 0.1, 1, 10]},
    "SVR": {"C": [0.1, 1, 10], "kernel": ["rbf"], "epsilon": [0.01, 0.1, 0.5]},
    "SVC": {"C": [0.1, 1, 10, 100], "kernel": ["rbf"], "gamma": ["scale", "auto"]},
    "LogisticRegression": {"C": [0.01, 0.1, 1, 10, 100], "penalty": ["l2"]},
    "KNeighborsClassifier": {"n_neighbors": [3, 5, 7, 9, 11], "weights": ["uniform", "distance"]},
    "KNeighborsRegressor": {"n_neighbors": [3, 5, 7, 9, 11], "weights": ["uniform", "distance"]},
    "RandomForestRegressor": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10, 20]},
    "RandomForestClassifier": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10, 20]},
    "GradientBoostingRegressor": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5]},
    "GradientBoostingClassifier": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [3, 5]},
}


# -------------------------------------------------
# HERO
# -------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-icon">🏠</div>
        <div class="hero-title">House Price Prediction Dashboard</div>
        <div class="hero-sub">Real Estate Analytics · Explore · Prepare · Train · Tune</div>
    </div>
    """,
    unsafe_allow_html=True,
)

labels = ["Setup", "EDA", "Clean", "Features", "Split", "Compare", "Train", "Metrics", "Tune"]
stage = st.session_state.pipeline_stage
rail_html = ""
for i, label in enumerate(labels):
    css_class = "done" if i < stage else ("active" if i == stage else "")
    rail_html += f'<div class="rail-step {css_class}">{label}</div>'

st.markdown(f'<div class="pipeline-rail">{rail_html}</div>', unsafe_allow_html=True)


# -------------------------------------------------
# TABS
# -------------------------------------------------
tabs = st.tabs([
    "🏠 Setup",
    "📊 EDA",
    "🧹 Clean",
    "🧩 Features",
    "✂️ Split",
    "📈 Compare",
    "🧠 Train",
    "📉 Metrics",
    "⚙️ Tune",
])


# -------------------------------------------------
# TAB 1 - SETUP
# -------------------------------------------------
with tabs[0]:
    step_badge(1, "Project Setup & Data Upload")

    left_col, right_col = st.columns([1, 2.4], gap="large")

    with left_col:
        st.markdown("#### Task Type")
        st.session_state.problem_type = st.radio(
            "Problem Type",
            options=["Regression", "Classification"],
            index=0 if st.session_state.problem_type == "Regression" else 1,
            label_visibility="collapsed",
        )
        st.markdown(f'<span class="small-pill">{st.session_state.problem_type}</span>', unsafe_allow_html=True)

    with right_col:
        st.markdown("#### Upload Housing Dataset (CSV)")
        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

        if uploaded is not None:
            df = pd.read_csv(uploaded)
            df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
            st.session_state.df = df.copy()
            if st.session_state.pipeline_stage == 0:
                st.session_state.pipeline_stage = 1
            st.success(f"Loaded {uploaded.name} successfully — {df.shape[0]:,} rows × {df.shape[1]} columns")

    if st.session_state.df is not None:
        df = st.session_state.df

        metric_cols = st.columns(4)
        metric_cols[0].metric("Total Records", f"{df.shape[0]:,}")
        metric_cols[1].metric("Total Features", f"{df.shape[1]}")
        metric_cols[2].metric("Numeric Features", f"{df.select_dtypes(include=np.number).shape[1]}")
        metric_cols[3].metric("Categorical Features", f"{df.select_dtypes(include=['object', 'category']).shape[1]}")

        st.markdown("#### Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True, height=300)

        col_target, col_pca = st.columns([1, 2], gap="large")

        with col_target:
            st.markdown("#### Target Variable")
            default_target = infer_default_target(df.columns.tolist())
            target_options = df.columns.tolist()
            default_index = target_options.index(default_target) if default_target in target_options else len(target_options) - 1
            st.session_state.target_col = st.selectbox(
                "Choose target column",
                target_options,
                index=default_index,
                label_visibility="collapsed",
            )
            st.markdown(
                f'<span class="small-pill">Target: {st.session_state.target_col}</span>',
                unsafe_allow_html=True,
            )

        with col_pca:
            st.markdown("#### PCA - 2D Housing Feature Map")
            numeric_features = df.drop(columns=[st.session_state.target_col], errors="ignore").select_dtypes(include=np.number).columns.tolist()
            selected_pca_features = st.multiselect(
                "Features for PCA",
                options=numeric_features,
                default=numeric_features[: min(6, len(numeric_features))],
                label_visibility="collapsed",
            )

            if st.button("Run PCA Projection"):
                if len(selected_pca_features) < 2:
                    st.warning("Select at least 2 numeric features for PCA.")
                else:
                    pca_input = df[selected_pca_features].dropna()
                    scaled = StandardScaler().fit_transform(pca_input)
                    components = PCA(n_components=2).fit_transform(scaled)
                    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])

                    if st.session_state.target_col in df.columns:
                        target_series = df.loc[pca_input.index, st.session_state.target_col]
                        pca_df["Target"] = target_series.values
                        fig = px.scatter(
                            pca_df,
                            x="PC1",
                            y="PC2",
                            color="Target",
                            color_continuous_scale=COLOR_SCALE,
                            opacity=0.7,
                            title="PCA Projection of Housing Data",
                        )
                    else:
                        fig = px.scatter(pca_df, x="PC1", y="PC2", title="PCA Projection of Housing Data")

                    fig.update_layout(**plot_theme(height=380))
                    fig.update_traces(marker=dict(size=5))
                    st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------
# TAB 2 - EDA
# -------------------------------------------------
with tabs[1]:
    step_badge(2, "Exploratory Data Analysis")

    if st.session_state.df is None:
        st.info("Please upload your housing dataset in the Setup tab first.")
    else:
        df = st.session_state.df
        numeric_df = df.select_dtypes(include=np.number)

        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.markdown("#### Descriptive Statistics")
            st.dataframe(df.describe(include="all").T, use_container_width=True, height=320)

        with c2:
            st.markdown("#### Missing Values Summary")
            missing = df.isnull().sum()
            missing_df = pd.DataFrame(
                {
                    "Column": missing.index,
                    "Missing": missing.values,
                    "Missing %": ((missing.values / len(df)) * 100).round(2),
                }
            )
            missing_df = missing_df[missing_df["Missing"] > 0]
            if missing_df.empty:
                st.success("No missing values found.")
            else:
                st.dataframe(missing_df, use_container_width=True, height=320)

        if not numeric_df.empty:
            st.markdown("#### Correlation Matrix")
            corr = numeric_df.corr(numeric_only=True)
            fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale=COLOR_SCALE)
            fig_corr.update_layout(**plot_theme(height=500))
            st.plotly_chart(fig_corr, use_container_width=True)

        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("#### Feature Distribution")
            dist_col = st.selectbox("Select feature", df.columns.tolist(), key="dist_feature")
            fig_hist = px.histogram(df, x=dist_col, nbins=50, marginal="box", color_discrete_sequence=[BLUE])
            fig_hist.update_layout(**plot_theme(height=360))
            st.plotly_chart(fig_hist, use_container_width=True)

        with right:
            st.markdown("#### Feature vs Target")
            target = st.session_state.target_col
            if target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
                possible_x = [col for col in numeric_df.columns if col != target]
                if possible_x:
                    x_feature = st.selectbox("Select X-axis feature", possible_x, key="scatter_feature")
                    fig_scatter = px.scatter(
                        df,
                        x=x_feature,
                        y=target,
                        opacity=0.45,
                        color_discrete_sequence=[TEAL],
                        title=f"{x_feature} vs {target}",
                    )
                    fig_scatter.update_layout(**plot_theme(height=360))
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info("No numeric feature available apart from target.")
            else:
                st.info("Select a numeric target column in Setup to view this chart.")

        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if categorical_cols:
            st.markdown("#### Categorical Feature Analysis")
            selected_cat = st.selectbox("Select categorical feature", categorical_cols, key="cat_feature")
            value_counts = df[selected_cat].astype(str).value_counts().reset_index()
            value_counts.columns = [selected_cat, "Count"]
            fig_bar = px.bar(value_counts, x=selected_cat, y="Count", color="Count", color_continuous_scale=COLOR_SCALE)
            fig_bar.update_layout(**plot_theme(height=350))
            st.plotly_chart(fig_bar, use_container_width=True)

            if st.session_state.target_col in df.columns and pd.api.types.is_numeric_dtype(df[st.session_state.target_col]):
                fig_box = px.box(df, x=selected_cat, y=st.session_state.target_col, color=selected_cat, color_discrete_sequence=PALETTE)
                fig_box.update_layout(**plot_theme(height=380, showlegend=False))
                st.plotly_chart(fig_box, use_container_width=True)


# -------------------------------------------------
# TAB 3 - CLEAN
# -------------------------------------------------
with tabs[2]:
    step_badge(3, "Data Cleaning & Preprocessing")

    if st.session_state.df is None:
        st.info("Please upload data first.")
    else:
        df = st.session_state.df
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        st.markdown("#### Missing Value Handling")
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                selected_col = st.selectbox("Column with missing values", missing_cols)
            with c2:
                strategy = st.selectbox("Imputation strategy", ["Mean", "Median", "Mode", "Drop Rows"])
            with c3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Apply Imputation"):
                    cleaned_df = st.session_state.df.copy()
                    if strategy == "Mean" and selected_col in numeric_cols:
                        cleaned_df[selected_col] = cleaned_df[selected_col].fillna(cleaned_df[selected_col].mean())
                    elif strategy == "Median" and selected_col in numeric_cols:
                        cleaned_df[selected_col] = cleaned_df[selected_col].fillna(cleaned_df[selected_col].median())
                    elif strategy == "Mode":
                        cleaned_df[selected_col] = cleaned_df[selected_col].fillna(cleaned_df[selected_col].mode()[0])
                    elif strategy == "Drop Rows":
                        cleaned_df = cleaned_df.dropna(subset=[selected_col]).reset_index(drop=True)
                    else:
                        st.warning("Mean and Median can only be used for numeric columns.")
                    st.session_state.df = cleaned_df
                    st.success(f"Applied {strategy} on {selected_col}.")
        else:
            st.success("No missing values found.")

        st.markdown("#### Drop Columns")
        available_drop_cols = [col for col in st.session_state.df.columns if col != st.session_state.target_col]
        drop_cols = st.multiselect("Select columns to drop", available_drop_cols)
        if st.button("Drop Selected Columns") and drop_cols:
            st.session_state.df = st.session_state.df.drop(columns=drop_cols)
            st.success(f"Dropped {len(drop_cols)} column(s).")

        st.markdown("#### Outlier Detection")
        current_numeric = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
        oc1, oc2 = st.columns([1, 2])
        with oc1:
            outlier_method = st.selectbox("Outlier method", ["None", "IQR", "Isolation Forest", "DBSCAN", "OPTICS"])
        with oc2:
            outlier_features = st.multiselect(
                "Features for outlier detection",
                current_numeric,
                default=current_numeric[: min(4, len(current_numeric))],
            )

        if outlier_method != "None" and outlier_features:
            base_df = st.session_state.df.copy()
            temp = base_df[outlier_features].dropna()
            outliers = pd.Series(False, index=temp.index)

            if outlier_method == "IQR":
                q1 = temp.quantile(0.25)
                q3 = temp.quantile(0.75)
                iqr = q3 - q1
                outliers = ((temp < (q1 - 1.5 * iqr)) | (temp > (q3 + 1.5 * iqr))).any(axis=1)
            elif outlier_method == "Isolation Forest":
                contamination = st.slider("Contamination", 0.01, 0.20, 0.05)
                preds = IsolationForest(contamination=contamination, random_state=42).fit_predict(temp)
                outliers = pd.Series(preds == -1, index=temp.index)
            elif outlier_method == "DBSCAN":
                eps = st.slider("eps", 0.1, 3.0, 0.7)
                preds = DBSCAN(eps=eps, min_samples=5).fit_predict(StandardScaler().fit_transform(temp))
                outliers = pd.Series(preds == -1, index=temp.index)
            elif outlier_method == "OPTICS":
                preds = OPTICS(min_samples=5).fit_predict(StandardScaler().fit_transform(temp))
                outliers = pd.Series(preds == -1, index=temp.index)

            count_outliers = int(outliers.sum())
            if count_outliers > 0:
                st.warning(f"Detected {count_outliers:,} outlier rows using {outlier_method}.")
                if len(outlier_features) >= 2:
                    vis_df = temp.copy()
                    vis_df["Type"] = np.where(outliers, "Outlier", "Normal")
                    fig_out = px.scatter(
                        vis_df,
                        x=outlier_features[0],
                        y=outlier_features[1],
                        color="Type",
                        color_discrete_map={"Normal": BLUE, "Outlier": RED},
                        opacity=0.6,
                    )
                    fig_out.update_layout(**plot_theme(height=360))
                    st.plotly_chart(fig_out, use_container_width=True)

                if st.button(f"Remove {count_outliers:,} Outlier Rows"):
                    st.session_state.df = st.session_state.df.drop(index=temp.index[outliers]).reset_index(drop=True)
                    st.success("Outlier rows removed successfully.")
            else:
                st.success("No outliers detected with the selected method.")

        st.markdown("#### Encode Categorical Features")
        categorical_cols = st.session_state.df.select_dtypes(include=["object", "category"]).columns.tolist()
        if categorical_cols:
            e1, e2 = st.columns([2, 1])
            with e1:
                encode_cols = st.multiselect("Columns to encode", categorical_cols, default=categorical_cols)
            with e2:
                encode_method = st.selectbox("Encoding method", ["Label Encoding", "One-Hot Encoding", "Ordinal Encoding"])

            if st.button("Encode Selected Columns"):
                encoded_df = st.session_state.df.copy()
                if encode_method == "Label Encoding":
                    for col in encode_cols:
                        le = LabelEncoder()
                        encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
                elif encode_method == "One-Hot Encoding":
                    encoded_df = pd.get_dummies(encoded_df, columns=encode_cols, drop_first=False)
                else:
                    oe = OrdinalEncoder()
                    encoded_df[encode_cols] = oe.fit_transform(encoded_df[encode_cols].astype(str))

                st.session_state.df = encoded_df
                st.success(f"{encode_method} applied successfully.")
        else:
            st.info("No categorical columns available for encoding.")

        st.markdown("#### Feature Scaling")
        scale_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
        if scale_cols:
            s1, s2 = st.columns([2, 1])
            with s1:
                cols_to_scale = st.multiselect("Columns to scale", scale_cols, default=scale_cols)
            with s2:
                scaler_name = st.selectbox("Scaler", ["StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler"])

            if st.button("Apply Scaling"):
                if cols_to_scale:
                    scaled_df = st.session_state.df.copy()
                    scaler = {
                        "StandardScaler": StandardScaler(),
                        "MinMaxScaler": MinMaxScaler(),
                        "RobustScaler": RobustScaler(),
                        "MaxAbsScaler": MaxAbsScaler(),
                    }[scaler_name]
                    scaled_df[cols_to_scale] = scaler.fit_transform(scaled_df[cols_to_scale])
                    st.session_state.df = scaled_df
                    st.success(f"{scaler_name} applied to {len(cols_to_scale)} column(s).")
                else:
                    st.warning("Please select at least one column to scale.")

        st.markdown("#### Current Dataset Preview")
        st.dataframe(st.session_state.df.head(8), use_container_width=True)


# -------------------------------------------------
# TAB 4 - FEATURES
# -------------------------------------------------
with tabs[3]:
    step_badge(4, "Feature Engineering & Selection")

    if st.session_state.df is None or st.session_state.target_col is None:
        st.info("Complete Setup and Cleaning first.")
    else:
        df = st.session_state.df.dropna().copy()
        target = st.session_state.target_col

        if target not in df.columns:
            st.error("Selected target column is not present in the current dataset.")
        else:
            numeric_feature_cols = [col for col in df.select_dtypes(include=np.number).columns if col != target]

            st.markdown("#### Create New Feature")
            if numeric_feature_cols:
                f1, f2, f3 = st.columns([2, 2, 1])
                with f1:
                    source_col = st.selectbox("Source column", numeric_feature_cols)
                with f2:
                    transform = st.selectbox(
                        "Transformation",
                        ["Log (log1p)", "Square", "Square Root", "Absolute", "Interaction (col1 × col2)"],
                    )
                with f3:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Add Feature"):
                        transformed_df = st.session_state.df.copy()
                        if transform == "Log (log1p)":
                            transformed_df[f"{source_col}_log"] = np.log1p(transformed_df[source_col].clip(lower=0))
                        elif transform == "Square":
                            transformed_df[f"{source_col}_sq"] = transformed_df[source_col] ** 2
                        elif transform == "Square Root":
                            transformed_df[f"{source_col}_sqrt"] = np.sqrt(transformed_df[source_col].clip(lower=0))
                        elif transform == "Absolute":
                            transformed_df[f"{source_col}_abs"] = transformed_df[source_col].abs()
                        elif transform == "Interaction (col1 × col2)":
                            other_cols = [col for col in numeric_feature_cols if col != source_col]
                            if other_cols:
                                other_col = other_cols[0]
                                transformed_df[f"{source_col}_x_{other_col}"] = transformed_df[source_col] * transformed_df[other_col]
                        st.session_state.df = transformed_df
                        st.success("New feature added successfully.")
            else:
                st.info("Numeric features are required for feature engineering.")

            st.markdown("#### Feature Selection")
            clean_df = st.session_state.df.dropna().copy()
            X = clean_df.drop(columns=[target])
            y = clean_df[target]
            X_numeric = X.select_dtypes(include=np.number)

            if X_numeric.empty:
                st.warning("No numeric features available. Encode categorical features first.")
            else:
                method = st.selectbox(
                    "Selection method",
                    ["Variance Threshold", "Correlation with Target", "Mutual Information"],
                )

                if method == "Variance Threshold":
                    threshold = st.slider("Minimum variance threshold", 0.0, 2.0, 0.05, step=0.01)
                    if st.button("Run Variance Threshold"):
                        selector = VarianceThreshold(threshold=threshold)
                        selector.fit(X_numeric)
                        kept = X_numeric.columns[selector.get_support()].tolist()
                        st.session_state.selected_features = kept

                        variance_df = pd.DataFrame({"Feature": X_numeric.columns, "Variance": X_numeric.var().values})
                        fig = px.bar(variance_df.sort_values("Variance", ascending=False), x="Feature", y="Variance", color="Variance", color_continuous_scale=COLOR_SCALE)
                        fig.update_layout(**plot_theme(height=360))
                        st.plotly_chart(fig, use_container_width=True)
                        st.success(f"Selected {len(kept)} features.")

                elif method == "Correlation with Target":
                    if not pd.api.types.is_numeric_dtype(y):
                        st.warning("Target must be numeric for correlation-based feature selection.")
                    else:
                        threshold = st.slider("Minimum absolute correlation", 0.0, 1.0, 0.3, step=0.01)
                        if st.button("Run Correlation Filter"):
                            correlations = pd.concat([X_numeric, y], axis=1).corr(numeric_only=True)[target].drop(target).abs()
                            kept = correlations[correlations >= threshold].index.tolist()
                            st.session_state.selected_features = kept

                            corr_df = correlations.sort_values(ascending=False).reset_index()
                            corr_df.columns = ["Feature", "Correlation"]
                            fig = px.bar(corr_df, x="Feature", y="Correlation", color="Correlation", color_continuous_scale=COLOR_SCALE)
                            fig.update_layout(**plot_theme(height=360))
                            st.plotly_chart(fig, use_container_width=True)
                            st.success(f"Selected {len(kept)} features.")

                else:
                    k = st.slider("Top-K features", 1, len(X_numeric.columns), min(8, len(X_numeric.columns)))
                    if st.button("Run Mutual Information"):
                        if st.session_state.problem_type == "Classification":
                            scores = mutual_info_classif(X_numeric, y, random_state=42)
                        else:
                            scores = mutual_info_regression(X_numeric, y, random_state=42)
                        mi_series = pd.Series(scores, index=X_numeric.columns).sort_values(ascending=False)
                        kept = mi_series.head(k).index.tolist()
                        st.session_state.selected_features = kept

                        mi_df = mi_series.reset_index()
                        mi_df.columns = ["Feature", "Score"]
                        fig = px.bar(mi_df, x="Feature", y="Score", color="Score", color_continuous_scale=COLOR_SCALE)
                        fig.update_layout(**plot_theme(height=360))
                        st.plotly_chart(fig, use_container_width=True)
                        st.success(f"Top {k} features selected.")

                if st.session_state.selected_features:
                    st.markdown(
                        f'<span class="small-pill">Selected Features: {len(st.session_state.selected_features)}</span>',
                        unsafe_allow_html=True,
                    )


# -------------------------------------------------
# TAB 5 - SPLIT
# -------------------------------------------------
with tabs[4]:
    step_badge(5, "Train-Test Split")

    if st.session_state.df is None or st.session_state.target_col is None:
        st.info("Complete the earlier steps first.")
    else:
        split_col1, split_col2, split_col3 = st.columns([2, 2, 1])
        with split_col1:
            test_ratio = st.slider("Test set percentage", 10, 40, 20) / 100
        with split_col2:
            random_seed = st.number_input("Random seed", value=42, step=1)
        with split_col3:
            stratify = st.checkbox("Stratify", value=False)

        if st.button("Run Split", use_container_width=True):
            split_df = st.session_state.df.dropna().copy()
            target = st.session_state.target_col

            if target not in split_df.columns:
                st.error("Target column not found in dataset.")
            else:
                if st.session_state.selected_features:
                    usable_features = [col for col in st.session_state.selected_features if col in split_df.columns]
                    X = split_df[usable_features]
                else:
                    X = split_df.drop(columns=[target]).select_dtypes(include=np.number)

                y = split_df[target]
                stratify_arg = y if (stratify and st.session_state.problem_type == "Classification") else None

                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=test_ratio,
                        random_state=int(random_seed),
                        stratify=stratify_arg,
                    )
                except Exception as exc:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=test_ratio,
                        random_state=int(random_seed),
                    )
                    st.warning(f"Stratification skipped: {exc}")

                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.pipeline_stage = max(st.session_state.pipeline_stage, 5)

                m1, m2, m3 = st.columns(3)
                m1.metric("Training Rows", f"{X_train.shape[0]:,}")
                m2.metric("Test Rows", f"{X_test.shape[0]:,}")
                m3.metric("Features Used", f"{X_train.shape[1]}")

                fig_pie = go.Figure(
                    go.Pie(
                        labels=["Train", "Test"],
                        values=[X_train.shape[0], X_test.shape[0]],
                        hole=0.6,
                        marker=dict(colors=[BLUE, ORANGE]),
                    )
                )
                fig_pie.update_layout(**plot_theme(height=320))
                st.plotly_chart(fig_pie, use_container_width=True)
                st.success("Train-test split completed successfully.")


# -------------------------------------------------
# TAB 6 - COMPARE
# -------------------------------------------------
with tabs[5]:
    step_badge(6, "Model Comparison")

    if st.session_state.X_train is None:
        st.info("Please run the Split step first.")
    else:
        model_pool = REGRESSION_MODELS if st.session_state.problem_type == "Regression" else CLASSIFICATION_MODELS

        c1, c2 = st.columns(2)
        with c1:
            cv_folds = st.slider("CV folds", 2, 10, 5)
        with c2:
            scale_features = st.checkbox("Scale features before comparison", value=True)

        selected_models = st.multiselect("Models to compare", list(model_pool.keys()), default=list(model_pool.keys()))

        if st.button("Run Model Comparison", use_container_width=True):
            X_train = st.session_state.X_train.values
            y_train = st.session_state.y_train.values

            if scale_features:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)

            scoring = "r2" if st.session_state.problem_type == "Regression" else "accuracy"
            results = []

            progress_bar = st.progress(0)
            for idx, model_name in enumerate(selected_models):
                model = model_pool[model_name]
                scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=KFold(n_splits=cv_folds, shuffle=True, random_state=42),
                    scoring=scoring,
                    n_jobs=-1,
                )
                results.append(
                    {
                        "Model": model_name,
                        "CV Mean": scores.mean(),
                        "CV Std": scores.std(),
                        "CV Min": scores.min(),
                        "CV Max": scores.max(),
                    }
                )
                progress_bar.progress((idx + 1) / len(selected_models))

            progress_bar.empty()
            result_df = pd.DataFrame(results).sort_values("CV Mean", ascending=False).reset_index(drop=True)
            st.session_state.model_comparison = result_df.to_dict()

            fig_compare = go.Figure()
            fig_compare.add_trace(
                go.Bar(
                    x=result_df["Model"],
                    y=result_df["CV Mean"],
                    error_y=dict(type="data", array=result_df["CV Std"], visible=True),
                    marker=dict(color=result_df["CV Mean"], colorscale=COLOR_SCALE),
                )
            )
            fig_compare.update_layout(**plot_theme(height=420, title=f"Cross-Validated {scoring.upper()} Comparison"))
            st.plotly_chart(fig_compare, use_container_width=True)

            st.dataframe(result_df, use_container_width=True)
            best_model_name = result_df.iloc[0]["Model"]
            st.success(f"Best model: {best_model_name} with mean CV score {result_df.iloc[0]['CV Mean']:.4f}")


# -------------------------------------------------
# TAB 7 - TRAIN
# -------------------------------------------------
with tabs[6]:
    step_badge(7, "Model Training & Evaluation")

    if st.session_state.X_train is None:
        st.info("Please run the Split step first.")
    else:
        model_pool = REGRESSION_MODELS if st.session_state.problem_type == "Regression" else CLASSIFICATION_MODELS

        t1, t2, t3 = st.columns([2, 2, 1])
        with t1:
            model_name = st.selectbox("Select model", list(model_pool.keys()))
        with t2:
            k_folds = st.number_input("K-fold splits", min_value=2, max_value=10, value=5)
        with t3:
            scale_data = st.checkbox("Scale", value=True)

        if st.button("Train Model", use_container_width=True):
            with st.spinner("Training model..."):
                model = model_pool[model_name]
                X_train = st.session_state.X_train.values
                X_test = st.session_state.X_test.values
                y_train = st.session_state.y_train.values
                y_test = st.session_state.y_test.values

                scaler = None
                if scale_data:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                scoring = "r2" if st.session_state.problem_type == "Regression" else "accuracy"
                cv_scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=KFold(n_splits=int(k_folds), shuffle=True, random_state=42),
                    scoring=scoring,
                    n_jobs=-1,
                )

                cv_df = pd.DataFrame({"Fold": range(1, int(k_folds) + 1), "Score": cv_scores})
                fig_cv = px.bar(cv_df, x="Fold", y="Score", color="Score", color_continuous_scale=COLOR_SCALE)
                fig_cv.update_layout(**plot_theme(height=300, title="Cross-Validation Scores"))
                st.plotly_chart(fig_cv, use_container_width=True)

                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                st.session_state.best_model = model
                st.session_state._scaler = scaler
                st.session_state._Xte_scaled = X_test
                st.session_state._yte = y_test
                st.session_state._Xtr_scaled = X_train
                st.session_state._ytr = y_train
                st.session_state.pipeline_stage = max(st.session_state.pipeline_stage, 7)

                if st.session_state.problem_type == "Regression":
                    train_r2 = r2_score(y_train, y_train_pred)
                    test_r2 = r2_score(y_test, y_test_pred)
                    mae = mean_absolute_error(y_test, y_test_pred)
                    model_rmse = rmse(y_test, y_test_pred)
                    model_mape = mape(y_test, y_test_pred)

                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Train R²", f"{train_r2:.4f}")
                    m2.metric("Test R²", f"{test_r2:.4f}")
                    m3.metric("MAE", f"{mae:,.2f}")
                    m4.metric("RMSE", f"{model_rmse:,.2f}")
                    m5.metric("MAPE", f"{model_mape:.2f}%")

                    fig_actual_pred = px.scatter(
                        x=y_test,
                        y=y_test_pred,
                        labels={"x": "Actual Price", "y": "Predicted Price"},
                        color_discrete_sequence=[BLUE],
                        title="Actual vs Predicted House Prices",
                    )
                    min_val = float(min(np.min(y_test), np.min(y_test_pred)))
                    max_val = float(max(np.max(y_test), np.max(y_test_pred)))
                    fig_actual_pred.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color=ORANGE, dash="dash"))
                    fig_actual_pred.update_layout(**plot_theme(height=400))
                    st.plotly_chart(fig_actual_pred, use_container_width=True)

                    residuals = y_test - y_test_pred
                    fig_resid = px.histogram(x=residuals, nbins=50, color_discrete_sequence=[TEAL], title="Residual Distribution")
                    fig_resid.update_layout(**plot_theme(height=300))
                    st.plotly_chart(fig_resid, use_container_width=True)

                    gap = train_r2 - test_r2
                    if gap > 0.15:
                        st.warning("The model may be overfitting because train performance is much higher than test performance.")
                    elif test_r2 < 0.50:
                        st.warning("The model may be underfitting. Try better features or another algorithm.")
                    else:
                        st.success("The model shows reasonable generalization.")

                else:
                    train_acc = accuracy_score(y_train, y_train_pred)
                    test_acc = accuracy_score(y_test, y_test_pred)
                    c1, c2 = st.columns(2)
                    c1.metric("Train Accuracy", f"{train_acc:.4f}")
                    c2.metric("Test Accuracy", f"{test_acc:.4f}")

                    cm = confusion_matrix(y_test, y_test_pred)
                    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale=COLOR_SCALE, title="Confusion Matrix")
                    fig_cm.update_layout(**plot_theme(height=380))
                    st.plotly_chart(fig_cm, use_container_width=True)

                    if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
                        probs = model.predict_proba(X_test)[:, 1]
                        auc_score = roc_auc_score(y_test, probs)
                        fpr, tpr, _ = roc_curve(y_test, probs)
                        fig_roc = go.Figure()
                        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={auc_score:.3f}", line=dict(color=BLUE, width=3)))
                        fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color=ORANGE, dash="dash"))
                        fig_roc.update_layout(**plot_theme(height=340, title="ROC Curve"))
                        st.plotly_chart(fig_roc, use_container_width=True)

                    report_df = pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True)).T
                    st.dataframe(report_df, use_container_width=True)

                if hasattr(model, "feature_importances_"):
                    feature_names = st.session_state.X_train.columns.tolist()
                    feature_importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
                    fi_df = feature_importance.reset_index()
                    fi_df.columns = ["Feature", "Importance"]
                    fig_fi = px.bar(fi_df, x="Feature", y="Importance", color="Importance", color_continuous_scale=COLOR_SCALE, title="Feature Importances")
                    fig_fi.update_layout(**plot_theme(height=360))
                    st.plotly_chart(fig_fi, use_container_width=True)

                export_object = {
                    "model": model,
                    "scaler": scaler,
                    "features": st.session_state.X_train.columns.tolist(),
                    "problem_type": st.session_state.problem_type,
                }
                st.markdown(
                    get_download_link(export_object, f"{model_name.replace(' ', '_').lower()}_house_price_model.pkl", "Download Trained Model"),
                    unsafe_allow_html=True,
                )


# -------------------------------------------------
# TAB 8 - PERFORMANCE METRICS
# -------------------------------------------------
with tabs[7]:
    step_badge(8, "Performance Metrics & Overfitting Analysis")

    if st.session_state.best_model is None:
        st.info("Please train a model first in the Train tab.")
    else:
        model = st.session_state.best_model
        X_train = st.session_state._Xtr_scaled
        X_test = st.session_state._Xte_scaled
        y_train = st.session_state._ytr
        y_test = st.session_state._yte
        problem_type = st.session_state.problem_type

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        if problem_type == "Classification":
            from sklearn.metrics import f1_score, precision_score, recall_score

            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)
            precision = precision_score(y_test, y_test_pred, average="weighted", zero_division=0)
            recall = recall_score(y_test, y_test_pred, average="weighted", zero_division=0)
            gap = train_acc - test_acc

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Train Accuracy", f"{train_acc:.4f}")
            k2.metric("Test Accuracy", f"{test_acc:.4f}")
            k3.metric("F1 Score", f"{f1:.4f}")
            k4.metric("Precision", f"{precision:.4f}")
            k5.metric("Recall", f"{recall:.4f}")

            if gap > 0.15:
                st.warning(f"Possible overfitting detected. Train/Test accuracy gap = {gap:.4f}")
            elif train_acc < 0.65:
                st.warning("Possible underfitting detected. Model performance is still low.")
            else:
                st.success("Model fit looks balanced on train and test data.")

            metric_tab1, metric_tab2, metric_tab3 = st.tabs(["Confusion Matrix", "ROC Curve", "Fit Analysis"])

            with metric_tab1:
                cm = confusion_matrix(y_test, y_test_pred)
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    color_continuous_scale=COLOR_SCALE,
                    title="Confusion Matrix",
                    labels={"x": "Predicted", "y": "Actual"},
                )
                fig_cm.update_layout(**plot_theme(height=380))
                st.plotly_chart(fig_cm, use_container_width=True)

            with metric_tab2:
                try:
                    if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
                        probs = model.predict_proba(X_test)[:, 1]
                        auc_score = roc_auc_score(y_test, probs)
                        fpr, tpr, _ = roc_curve(y_test, probs)
                        fig_roc = px.area(
                            x=fpr,
                            y=tpr,
                            title=f"ROC Curve (AUC = {auc_score:.4f})",
                            labels={"x": "False Positive Rate", "y": "True Positive Rate"},
                        )
                        fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color=ORANGE, dash="dash"))
                        fig_roc.update_layout(**plot_theme(height=360))
                        st.plotly_chart(fig_roc, use_container_width=True)
                    else:
                        st.info("ROC curve is shown only for binary classification models with predict_proba support.")
                except Exception as exc:
                    st.warning(f"ROC curve not available: {exc}")

            with metric_tab3:
                fit_df = pd.DataFrame({
                    "Dataset": ["Train Accuracy", "Test Accuracy"],
                    "Score": [train_acc, test_acc],
                })
                fig_fit = px.bar(fit_df, x="Dataset", y="Score", color="Dataset", color_discrete_sequence=[BLUE, ORANGE], title="Train vs Test Accuracy")
                fig_fit.update_layout(**plot_theme(height=340, showlegend=False))
                st.plotly_chart(fig_fit, use_container_width=True)

        else:
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            test_rmse = rmse(y_test, y_test_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_mape = mape(y_test, y_test_pred)
            gap = train_r2 - test_r2

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Train R²", f"{train_r2:.4f}")
            k2.metric("Test R²", f"{test_r2:.4f}")
            k3.metric("RMSE", f"{test_rmse:,.2f}")
            k4.metric("MAE", f"{test_mae:,.2f}")
            k5.metric("MAPE", f"{test_mape:.2f}%")

            if gap > 0.15:
                st.warning(f"Possible overfitting detected. Train/Test R² gap = {gap:.4f}")
            elif test_r2 < 0.40:
                st.warning("Possible underfitting detected. Test R² is still low.")
            else:
                st.success("Model fit looks balanced on train and test data.")

            metric_tab1, metric_tab2, metric_tab3 = st.tabs(["Actual vs Predicted", "Residuals", "Fit Analysis"])

            with metric_tab1:
                fig_ap = px.scatter(
                    x=y_test,
                    y=y_test_pred,
                    labels={"x": "Actual Price", "y": "Predicted Price"},
                    title="Actual vs Predicted House Prices",
                    color_discrete_sequence=[TEAL],
                    opacity=0.65,
                )
                min_val = float(min(np.min(y_test), np.min(y_test_pred)))
                max_val = float(max(np.max(y_test), np.max(y_test_pred)))
                fig_ap.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color=RED, dash="dash"))
                fig_ap.update_layout(**plot_theme(height=400))
                st.plotly_chart(fig_ap, use_container_width=True)

            with metric_tab2:
                residuals = np.array(y_test) - np.array(y_test_pred)
                fig_res = px.histogram(
                    x=residuals,
                    nbins=50,
                    title="Residual Distribution",
                    color_discrete_sequence=[PURPLE],
                )
                fig_res.add_vline(x=0, line_dash="dash", line_color=ORANGE)
                fig_res.update_layout(**plot_theme(height=340))
                st.plotly_chart(fig_res, use_container_width=True)

            with metric_tab3:
                fit_df = pd.DataFrame({
                    "Dataset": ["Train R²", "Test R²"],
                    "Score": [train_r2, test_r2],
                })
                fig_fit = px.bar(fit_df, x="Dataset", y="Score", color="Dataset", color_discrete_sequence=[BLUE, ORANGE], title="Train vs Test R²")
                fig_fit.update_layout(**plot_theme(height=340, showlegend=False))
                st.plotly_chart(fig_fit, use_container_width=True)


# -------------------------------------------------
# TAB 9 - TUNE
# -------------------------------------------------
with tabs[8]:
    step_badge(8, "Hyperparameter Tuning")

    if st.session_state.best_model is None:
        st.info("Please train a model first in the Train tab.")
    else:
        model = st.session_state.best_model
        model_type = type(model).__name__
        st.markdown(f'<span class="small-pill">Current Model: {model_type}</span>', unsafe_allow_html=True)

        if model_type not in GRIDS or not GRIDS[model_type]:
            st.warning(f"No tuning grid configured for {model_type}.")
        else:
            st.json(GRIDS[model_type])

            u1, u2 = st.columns(2)
            with u1:
                search_method = st.radio("Search method", ["Grid Search", "Random Search"])
            with u2:
                tune_cv = st.number_input("CV folds", min_value=2, max_value=10, value=3)
                n_iter = 20
                if search_method == "Random Search":
                    n_iter = st.number_input("Random iterations", min_value=5, max_value=100, value=20)

            if st.button("Start Tuning", use_container_width=True):
                with st.spinner("Tuning hyperparameters..."):
                    X_train = st.session_state._Xtr_scaled
                    y_train = st.session_state._ytr
                    scoring = "r2" if st.session_state.problem_type == "Regression" else "accuracy"

                    if search_method == "Grid Search":
                        searcher = GridSearchCV(model, GRIDS[model_type], cv=int(tune_cv), scoring=scoring, n_jobs=-1)
                    else:
                        searcher = RandomizedSearchCV(model, GRIDS[model_type], cv=int(tune_cv), n_iter=int(n_iter), random_state=42, scoring=scoring, n_jobs=-1)

                    searcher.fit(X_train, y_train)

                st.success("Hyperparameter tuning completed.")
                st.metric("Best CV Score", f"{searcher.best_score_:.4f}")
                st.write("Best Parameters:")
                st.json(searcher.best_params_)

                results_df = pd.DataFrame(searcher.cv_results_)
                top_df = results_df.nlargest(min(20, len(results_df)), "mean_test_score").reset_index(drop=True)
                top_df["Rank"] = top_df.index.astype(str)

                fig_tune = px.bar(top_df, x="Rank", y="mean_test_score", color="mean_test_score", color_continuous_scale=COLOR_SCALE, title="Top Tuning Results")
                fig_tune.update_layout(**plot_theme(height=350))
                st.plotly_chart(fig_tune, use_container_width=True)

                best_estimator = searcher.best_estimator_
                y_test_true = st.session_state._yte
                y_test_pred = best_estimator.predict(st.session_state._Xte_scaled)
                st.session_state.tuned_model = best_estimator

                if st.session_state.problem_type == "Regression":
                    tuned_r2 = r2_score(y_test_true, y_test_pred)
                    tuned_rmse = rmse(y_test_true, y_test_pred)
                    tuned_mae = mean_absolute_error(y_test_true, y_test_pred)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Tuned R²", f"{tuned_r2:.4f}")
                    m2.metric("Tuned RMSE", f"{tuned_rmse:,.2f}")
                    m3.metric("Tuned MAE", f"{tuned_mae:,.2f}")
                else:
                    tuned_acc = accuracy_score(y_test_true, y_test_pred)
                    st.metric("Tuned Accuracy", f"{tuned_acc:.4f}")

                tuned_export = {
                    "model": best_estimator,
                    "scaler": st.session_state._scaler,
                    "features": st.session_state.X_train.columns.tolist(),
                    "best_params": searcher.best_params_,
                    "problem_type": st.session_state.problem_type,
                }
                st.markdown(
                    get_download_link(tuned_export, f"tuned_{model_type.lower()}_house_price_model.pkl", "Download Tuned Model"),
                    unsafe_allow_html=True,
                )


# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <p style='text-align:center;color:#8aa4bf;font-size:12px;'>
        🏠 House Price Prediction Dashboard · Built with Streamlit, scikit-learn and Plotly
    </p>
    """,
    unsafe_allow_html=True,
)
