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
# Light styling -- a calm, real-estate-appropriate palette
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #F7F5F0;
        border: 1px solid #E4DFD3;
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.6rem;
    }
    .price-result {
        background-color: #EAF3EC;
        border: 1px solid #B9DAC2;
        border-radius: 12px;
        padding: 1.6rem;
        text-align: center;
    }
    .price-result h1 {
        color: #1B4332;
        font-size: 2.6rem;
        margin: 0;
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
    st.title("Ames Housing Price Prediction")
    st.write(
        "This application predicts the **sale price of a house** in Ames, Iowa, "
        "based on its physical characteristics, quality ratings, and location."
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
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Project Information**")
        st.write(f"**Problem Type:** Regression")
        st.write(f"**Dataset:** AmesHousing.csv")
        st.write(f"**Target:** SalePrice")
        model_name = bundle["model_name"] if bundle else "Not trained yet"
        st.write(f"**Model:** {model_name}")
        if bundle:
            st.write(f"**Test R²:** {bundle['metrics']['test']['r2']:.3f}")
        st.markdown("</div>", unsafe_allow_html=True)

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
                        <p style="margin:0; color:#3A5A40; font-size:1.1rem;">Estimated House Price</p>
                        <h1>${prediction:,.0f}</h1>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Something went wrong while generating the prediction: {e}")
