"""
app.py
------
Ames Housing Price Prediction — Streamlit application.

Uses the pipeline + metadata saved by train.py (model/model.pkl). Does not
retrain anything at startup.
"""

import os

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

import features as feat

DATA_PATH = "AmesHousing.csv"
MODEL_PATH = os.path.join("model", "model.pkl")

st.set_page_config(
    page_title="Ames Housing Price Prediction",
    page_icon="\U0001F3E0",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Styling -- warm brass/clay "real estate ledger" palette on a dark base
# (paired with .streamlit/config.toml, which sets the native theme colors)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class^="css"], [class*=" css"] { font-family: 'Inter', sans-serif; }

    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3 {
        font-family: 'Fraunces', serif;
        letter-spacing: 0.01em;
    }
    [data-testid="stAppViewContainer"] h1 {
        border-bottom: 3px solid #B0522A;
        padding-bottom: 0.45rem;
        display: inline-block;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #14120D 0%, #1B1811 100%);
        border-right: 1px solid #3A3226;
    }
    section[data-testid="stSidebar"] h1 {
        font-family: 'Fraunces', serif;
        color: #C9973E;
        font-size: 1.4rem;
        border-bottom: 2px solid #C9973E;
        padding-bottom: 0.6rem;
        margin-bottom: 1rem;
    }

    /* Hero (Home page) */
    .hero {
        background: radial-gradient(circle at 85% -10%, rgba(201,151,62,0.16), transparent 55%),
                    linear-gradient(135deg, #241D14 0%, #17140F 75%);
        border: 1px solid #3A3226;
        border-radius: 18px;
        padding: 2.6rem 3rem;
        margin-bottom: 1.6rem;
    }
    .hero-title {
        font-family: 'Fraunces', serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: #F1EAD9;
        margin: 0 0 0.6rem 0;
        line-height: 1.08;
    }
    .hero-sub {
        color: #B8AD98;
        font-size: 1.08rem;
        max-width: 640px;
        margin: 0;
    }
    .hero-rule {
        width: 60px; height: 3px; background: #B0522A; border-radius: 2px;
        margin: 1.1rem 0 1.2rem 0;
    }

    /* Info / metric cards */
    .info-card {
        background: #201C16;
        border: 1px solid #3A3226;
        border-left: 4px solid #C9973E;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
    }
    div[data-testid="stMetric"] {
        background: #201C16;
        border: 1px solid #3A3226;
        border-radius: 12px;
        padding: 0.9rem 1rem 0.6rem 1rem;
    }
    div[data-testid="stMetricValue"] { color: #C9973E; }
    div[data-testid="stMetricLabel"] { color: #B8AD98; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #B0522A, #8C3D1E);
        color: #F5EFE6;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.65rem 1.3rem;
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 18px rgba(176, 82, 42, 0.35);
        color: #F5EFE6;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #3A3226; gap: 6px; }
    .stTabs [data-baseweb="tab"] { color: #B8AD98; font-weight: 500; }
    .stTabs [aria-selected="true"] {
        color: #C9973E !important;
        border-bottom: 2px solid #C9973E !important;
    }

    /* Prediction result */
    .price-result {
        background: linear-gradient(135deg, #22321F, #1B2A18);
        border: 1px solid #3E5A3A;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .price-result .price-label {
        color: #9FC79A;
        font-size: 1.05rem;
        letter-spacing: 0.01em;
        margin: 0;
    }
    .price-result .price-amount {
        font-family: 'Fraunces', serif;
        color: #EFF7EC;
        font-size: 3rem;
        font-weight: 700;
        margin: 0.25rem 0 0 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        return None
    df_raw = pd.read_csv(DATA_PATH)
    df = feat.normalize_columns(df_raw)
    df = df.drop(columns=["Order", "PID"], errors="ignore")
    return df


@st.cache_resource
def load_model_bundle():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


df = load_dataset()
bundle = load_model_bundle()


def predict_price(raw_input_df, bundle):
    """Apply the exact same manual preprocessing used in train.py, then the
    saved pipeline, to a raw single-row (or multi-row) input DataFrame."""
    X = feat.prepare_base_features(raw_input_df)
    X = feat.apply_skew_transform(X, bundle["skewed_cols"])
    X = X[bundle["feature_columns"]]
    pred_log = bundle["pipeline"].predict(X)
    return np.expm1(pred_log)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Ames Housing ML")
page = st.sidebar.radio(
    "Navigate",
    [
        "\U0001F3E0 Home",
        "\U0001F4CA Data Overview",
        "\U0001F4C8 EDA",
        "\U0001F916 Model Performance",
        "\U0001F52E House Price Prediction",
    ],
)

if df is None:
    st.sidebar.error("AmesHousing.csv not found next to app.py.")
if bundle is None:
    st.sidebar.warning("model/model.pkl not found. Run `python train.py` first.")


# ---------------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------------
if page.endswith("Home"):
    st.markdown(
        """
        <div class="hero">
            <p class="hero-title">Ames Housing Price Prediction</p>
            <div class="hero-rule"></div>
            <p class="hero-sub">
                Predicting what a home in Ames, Iowa is worth — from its size and quality
                to its neighborhood — using a regression model trained on real sale records.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("What this project does")
        st.markdown(
            """
- Uses the **AmesHousing.csv** dataset (individual residential property sales in Ames, Iowa).
- Frames house price prediction as a **regression** problem: given a house's features, estimate `SalePrice`.
- Trains and compares several regression models, then selects the one that generalizes best on a held-out validation set.
- Applies that trained pipeline here so you can get an estimated price for a house you describe yourself.

**Why machine learning helps here:** a house's price depends on dozens of
interacting factors — size, quality, age, neighborhood, amenities — in ways
that are hard to capture with a simple rule of thumb. A regression model can
learn these relationships directly from thousands of historical sales.
            """
        )
    with col2:
        model_name = bundle["model_name"] if bundle else "Not trained yet"
        test_r2_line = f"<p style='margin:0.3rem 0;'><b>Test R²:</b> {bundle['metrics']['test']['r2']:.3f}</p>" if bundle else ""
        st.markdown(
            f"""
            <div class="info-card">
                <p style="margin:0 0 0.6rem 0; font-family:'Fraunces',serif; font-size:1.05rem;">Project Information</p>
                <p style="margin:0.3rem 0;"><b>Problem Type:</b> Regression</p>
                <p style="margin:0.3rem 0;"><b>Dataset:</b> AmesHousing.csv</p>
                <p style="margin:0.3rem 0;"><b>Target:</b> SalePrice</p>
                <p style="margin:0.3rem 0;"><b>Model:</b> {model_name}</p>
                {test_r2_line}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if df is None:
        st.error("AmesHousing.csv not found. Place it next to app.py.")
    if bundle is None:
        st.error("Model file not found. Please run `python train.py` first.")


# ---------------------------------------------------------------------------
# DATA OVERVIEW
# ---------------------------------------------------------------------------
elif page.endswith("Data Overview"):
    st.title("Data Overview")

    if df is None:
        st.error("AmesHousing.csv not found. Please place the dataset next to app.py.")
    else:
        st.subheader("Dataset Statistics")
        n_num = df.select_dtypes(include=[np.number]).shape[1]
        n_cat = df.shape[1] - n_num
        total_missing = int(df.isnull().sum().sum())

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", f"{df.shape[1]:,}")
        c3.metric("Numerical features", n_num)
        c4.metric("Categorical features", n_cat)
        c5.metric("Missing values", f"{total_missing:,}")

        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Data Types")
        dtypes_df = pd.DataFrame({"Column": df.columns, "Dtype": df.dtypes.astype(str).values})
        st.dataframe(dtypes_df, use_container_width=True, height=300)

        st.subheader("Missing Values")
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if len(missing) > 0:
            missing_df = pd.DataFrame(
                {"Column": missing.index, "Missing Count": missing.values,
                 "Missing %": (missing.values / len(df) * 100).round(2)}
            )
            st.dataframe(missing_df, use_container_width=True, height=300)
        else:
            st.info("No missing values in this dataset.")

        st.subheader("Statistical Summary")
        st.dataframe(df.describe().T, use_container_width=True)


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
elif page.endswith("EDA"):
    st.title("Exploratory Data Analysis")

    if df is None:
        st.error("AmesHousing.csv not found. Please place the dataset next to app.py.")
    else:
        target = "SalePrice"

        st.subheader("Sale Price Distribution")
        fig = px.histogram(df, x=target, nbins=50, marginal="box",
                            title="Distribution of SalePrice")
        fig.update_layout(bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr(numeric_only=True)
        if target in corr.columns:
            top_features = corr[target].abs().sort_values(ascending=False).head(15).index
        else:
            top_features = corr.columns[:15]
        corr_subset = numeric_df[top_features].corr()
        fig2 = px.imshow(
            corr_subset, text_auto=".2f", color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, aspect="auto",
            title="Correlation — top numerical features vs SalePrice",
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Area vs Sale Price")
        if "GrLivArea" in df.columns:
            fig3 = px.scatter(
                df, x="GrLivArea", y=target, opacity=0.5,
                title="Above-Ground Living Area vs SalePrice",
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("GrLivArea column not found in this dataset.")

        st.subheader("Important Numerical Features")
        numeric_cols_for_select = [c for c in numeric_df.columns if c != target]
        default_idx = numeric_cols_for_select.index("OverallQual") if "OverallQual" in numeric_cols_for_select else 0
        chosen_num = st.selectbox("Choose a numerical feature", numeric_cols_for_select, index=default_idx)
        fig4 = px.scatter(df, x=chosen_num, y=target, opacity=0.5,
                           title=f"{chosen_num} vs SalePrice")
        st.plotly_chart(fig4, use_container_width=True)

        st.subheader("Categorical Feature Analysis")
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if cat_cols:
            default_cat_idx = cat_cols.index("Neighborhood") if "Neighborhood" in cat_cols else 0
            chosen_cat = st.selectbox("Choose a categorical feature", cat_cols, index=default_cat_idx)
            fig5 = px.box(df, x=chosen_cat, y=target, title=f"SalePrice by {chosen_cat}")
            fig5.update_xaxes(tickangle=45)
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("No categorical columns found in this dataset.")


# ---------------------------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------------------------
elif page.endswith("Model Performance"):
    st.title("Model Performance")

    if bundle is None:
        st.error("Model file not found. Please run `python train.py` first.")
    else:
        metrics = bundle["metrics"]
        st.caption(f"Selected model: **{bundle['model_name']}**")

        st.subheader("Regression Metrics (Test Set)")
        c1, c2, c3 = st.columns(3)
        c1.metric("R²", f"{metrics['test']['r2']:.4f}")
        c2.metric("MAE", f"${metrics['test']['mae']:,.0f}")
        c3.metric("RMSE", f"${metrics['test']['rmse']:,.0f}")

        with st.expander("Train / Validation / Test comparison"):
            split_metrics = {k: v for k, v in metrics.items() if k in ("train", "validation", "test")}
            comp_df = pd.DataFrame(split_metrics).T[["r2", "mae", "rmse"]]
            comp_df.columns = ["R²", "MAE", "RMSE"]
            comp_df["MAE"] = comp_df["MAE"].map(lambda v: f"${v:,.0f}")
            comp_df["RMSE"] = comp_df["RMSE"].map(lambda v: f"${v:,.0f}")
            comp_df["R²"] = comp_df["R²"].map(lambda v: f"{v:.4f}")
            st.dataframe(comp_df, use_container_width=True)

        y_actual = bundle["y_test_actual"]
        y_pred = bundle["y_test_pred_actual"]

        st.subheader("Actual vs Predicted Plot")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_actual, y=y_pred, mode="markers",
                                  opacity=0.6, name="Test houses"))
        lims = [min(y_actual.min(), y_pred.min()), max(y_actual.max(), y_pred.max())]
        fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                                  line=dict(dash="dash", color="red"), name="Perfect prediction"))
        fig.update_layout(xaxis_title="Actual Sale Price ($)", yaxis_title="Predicted Sale Price ($)",
                           title="Actual vs Predicted Sale Price (Test Set)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Residual Plot")
        residuals = y_pred - y_actual
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers", opacity=0.6))
        fig2.add_hline(y=0, line_dash="dash", line_color="red")
        fig2.update_layout(xaxis_title="Predicted Price ($)", yaxis_title="Residual (Predicted - Actual)",
                            title="Residuals vs Predicted Price")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            "A residual is the difference between the predicted and actual sale price. "
            "Residuals scattered randomly around zero, with no clear pattern, indicate the "
            "model's errors are not systematically biased for any price range."
        )


# ---------------------------------------------------------------------------
# HOUSE PRICE PREDICTION
# ---------------------------------------------------------------------------
elif page.endswith("House Price Prediction"):
    st.title("House Price Prediction")

    if bundle is None or df is None:
        st.error("Both AmesHousing.csv and model/model.pkl are required for predictions. "
                  "Please add the dataset and run `python train.py` first.")
    else:
        numerical_cols = bundle["numerical_cols"]
        ordinal_cols = bundle["ordinal_cols"]
        nominal_cols = bundle["nominal_cols"]

        st.write("Enter the house's characteristics below, then click **Predict House Price**.")

        user_input = {}

        tab_num, tab_ord, tab_nom = st.tabs(
            ["Size & Numerical", "Quality & Condition", "Type & Location"]
        )

        with tab_num:
            cols = st.columns(3)
            for i, col_name in enumerate(numerical_cols):
                series = df[col_name].dropna()
                if len(series) == 0:
                    default_val, min_val, max_val = 0.0, 0.0, 100.0
                else:
                    default_val = float(series.median())
                    min_val = float(series.min())
                    max_val = float(series.max())
                is_int_like = pd.api.types.is_integer_dtype(df[col_name]) or float(default_val).is_integer()
                with cols[i % 3]:
                    if is_int_like:
                        user_input[col_name] = st.number_input(
                            col_name, min_value=int(min_val), max_value=int(max_val * 2 if max_val > 0 else 10),
                            value=int(default_val), step=1, key=f"num_{col_name}",
                        )
                    else:
                        user_input[col_name] = st.number_input(
                            col_name, min_value=min_val, value=default_val, step=1.0, key=f"num_{col_name}",
                        )

        with tab_ord:
            cols = st.columns(3)
            for i, col_name in enumerate(ordinal_cols):
                categories = feat.ORDINAL_FEATURES[col_name]
                default_idx = len(categories) // 2
                with cols[i % 3]:
                    user_input[col_name] = st.selectbox(
                        col_name, categories, index=default_idx, key=f"ord_{col_name}",
                    )

        with tab_nom:
            cols = st.columns(3)
            for i, col_name in enumerate(nominal_cols):
                options = sorted(df[col_name].dropna().unique().tolist(), key=lambda x: str(x))
                if not options:
                    options = ["Missing"]
                with cols[i % 3]:
                    user_input[col_name] = st.selectbox(
                        col_name, options, index=0, key=f"nom_{col_name}",
                    )

        st.markdown("---")

        if st.button("\U0001F52E Predict House Price", type="primary", use_container_width=True):
            try:
                input_df = pd.DataFrame([user_input])
                prediction = predict_price(input_df, bundle)[0]
                st.markdown(
                    f"""
                    <div class="price-result">
                        <p class="price-label">Estimated House Price</p>
                        <p class="price-amount">${prediction:,.0f}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Something went wrong while generating the prediction: {e}")
