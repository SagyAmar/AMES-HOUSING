"""
train.py
--------
Ames Housing Price Prediction — training script.

This reproduces the existing ML project's workflow (see the accompanying
notebook / README) without changing its preprocessing or modeling logic:

  1. Load AmesHousing.csv, normalize column names, drop identifier columns.
  2. Fill "NA-means-no-feature" columns, add the 4 engineered numerical
     features (TotalSF, HouseAge, RemodAge, TotalBaths).
  3. log1p-transform the target (SalePrice), then split 60/20/20 into
     train/validation/test (random_state=42).
  4. Remove the known Ames outliers (GrLivArea > 4000 & price < $300k) from
     the TRAIN split only.
  5. Log1p-transform numerical columns that are skewed on the TRAIN split
     only (|skew| > 0.75), applying the same column list to val/test.
  6. Build the ColumnTransformer (median-impute+scale numerical, impute+
     ordinal-encode+scale ordinal, impute+one-hot nominal).
  7. Cross-validate 5 baseline models, then GridSearchCV-tune Ridge and SVR,
     and pick whichever tuned model has the lower validation RMSE (never the
     test set) -- exactly the notebook's selection protocol.
  8. Evaluate the selected model once on the held-out test set.
  9. Save the fitted pipeline + all metadata the Streamlit app needs
     (skewed-column list, metrics, test-set predictions for the diagnostic
     plots) to model/model.pkl with joblib.

Run with:  python train.py
"""

import os
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import features as feat

DATA_PATH = "AmesHousing.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")


def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Place AmesHousing.csv next to train.py before running this script."
        )
    df_raw = pd.read_csv(path)
    df_raw = feat.normalize_columns(df_raw)
    # Drop identifier columns (no predictive value) -- same as the notebook.
    df = df_raw.drop(columns=["Order", "PID"], errors="ignore").copy()
    return df


def build_preprocessor(num_cols, ordinal_cols, nominal_cols):
    ordinal_categories = [feat.ORDINAL_FEATURES[col] for col in ordinal_cols]

    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    ord_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("ordinal_encoder", OrdinalEncoder(
            categories=ordinal_categories, handle_unknown="use_encoded_value", unknown_value=-1
        )),
        ("scaler", StandardScaler()),
    ])

    nom_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, num_cols),
        ("ord", ord_pipeline, ordinal_cols),
        ("nom", nom_pipeline, nominal_cols),
    ])
    return preprocessor


def eval_dollar(pipe, X, y_log):
    pred = np.expm1(pipe.predict(X))
    actual = np.expm1(y_log)
    mae = mean_absolute_error(actual, pred)
    mse = mean_squared_error(actual, pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(actual, pred)
    return float(mae), float(mse), rmse, float(r2)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading data...")
    df = load_data()
    df = feat.prepare_base_features(df)  # NA-sentinel fill + engineered cols

    target = feat.TARGET
    ordinal_cols = feat.ORDINAL_COLS
    nominal_cols = feat.NOMINAL_COLS
    numerical_cols = feat.get_numerical_cols(df.drop(columns=feat.ENGINEERED_NUM_COLS))
    num_cols = numerical_cols + feat.ENGINEERED_NUM_COLS

    print(f"Numerical features: {len(num_cols)}  Ordinal: {len(ordinal_cols)}  Nominal: {len(nominal_cols)}")

    y_full = np.log1p(df[target])
    X_full = df.drop(columns=[target])

    # 60% train / 20% validation / 20% test (same split as the notebook)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=feat.RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=feat.RANDOM_STATE
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # Outlier removal -- TRAIN SET ONLY (known Ames data quirk)
    if "GrLivArea" in X_train.columns:
        outlier_mask = (X_train["GrLivArea"] > 4000) & (np.expm1(y_train) < 300000)
        print(f"Removing {int(outlier_mask.sum())} outlier rows from the TRAIN set only")
        X_train = X_train[~outlier_mask]
        y_train = y_train[~outlier_mask]

    # Skew correction -- decided on TRAIN ONLY, applied identically everywhere
    skewed_cols = feat.compute_skewed_cols(X_train, num_cols)
    print(f"Log-transforming {len(skewed_cols)} skewed numerical columns: {skewed_cols}")
    X_train = feat.apply_skew_transform(X_train, skewed_cols)
    X_val = feat.apply_skew_transform(X_val, skewed_cols)
    X_test = feat.apply_skew_transform(X_test, skewed_cols)

    preprocessor = build_preprocessor(num_cols, ordinal_cols, nominal_cols)

    # --- Baseline models + 5-fold CV ---
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0),
        "Lasso Regression": Lasso(alpha=0.001),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
        "SVR": SVR(C=1.0, kernel="rbf"),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=feat.RANDOM_STATE)
    print("\n--- Cross-Validation (RMSE, log scale) ---")
    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        scores = cross_val_score(pipe, X_train, y_train, scoring="neg_root_mean_squared_error", cv=kf, n_jobs=-1)
        rmse_scores = -scores
        print(f"{name}: Mean RMSE = {rmse_scores.mean():.4f} (+/- {rmse_scores.std():.4f})")

    # --- Hyperparameter tuning: Ridge + SVR (the two strongest baselines) ---
    param_grids = {
        "Ridge Regression": {
            "model": Ridge(),
            "param_grid": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 20.0, 50.0]},
        },
        "SVR": {
    "model": SVR(),
    "param_grid": {
        "model__C": [1, 10],
        "model__gamma": ["scale", 0.01],
        "model__kernel": ["rbf"],
    },
},}

    print("\n--- Hyperparameter Tuning ---")
    tuned_results = {}
    for name, cfg in param_grids.items():
        tune_pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", cfg["model"])])
        grid_search = GridSearchCV(
            tune_pipe, cfg["param_grid"], cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        val_pred_log = grid_search.best_estimator_.predict(X_val)
        val_rmse_log = float(np.sqrt(mean_squared_error(y_val, val_pred_log)))
        tuned_results[name] = {
            "best_estimator": grid_search.best_estimator_,
            "best_params": grid_search.best_params_,
            "val_rmse_log": val_rmse_log,
        }
        print(f"{name}: best params = {grid_search.best_params_}  Val RMSE (log) = {val_rmse_log:.4f}")

    # Select whichever tuned model does best on validation (never test)
    best_name = min(tuned_results, key=lambda n: tuned_results[n]["val_rmse_log"])
    best_model = tuned_results[best_name]["best_estimator"]
    print(f"\nSelected model: {best_name}")

    # --- Final, one-time evaluation on the held-out test set ---
    train_mae, train_mse, train_rmse, train_r2 = eval_dollar(best_model, X_train, y_train)
    val_mae, val_mse, val_rmse, val_r2 = eval_dollar(best_model, X_val, y_val)
    test_mae, test_mse, test_rmse, test_r2 = eval_dollar(best_model, X_test, y_test)

    print(f"\nTest set  -> R2: {test_r2:.4f}  MAE: ${test_mae:,.2f}  RMSE: ${test_rmse:,.2f}")

    y_test_actual = np.expm1(y_test).to_numpy()
    y_test_pred_actual = np.expm1(best_model.predict(X_test))

    metrics = {
        "model_name": best_name,
        "best_params": tuned_results[best_name]["best_params"],
        "train": {"r2": train_r2, "mae": train_mae, "rmse": train_rmse},
        "validation": {"r2": val_r2, "mae": val_mae, "rmse": val_rmse},
        "test": {"r2": test_r2, "mae": test_mae, "rmse": test_rmse},
    }

    bundle = {
        "pipeline": best_model,
        "model_name": best_name,
        "metrics": metrics,
        "skewed_cols": skewed_cols,
        "num_cols": num_cols,
        "numerical_cols": numerical_cols,
        "ordinal_cols": ordinal_cols,
        "nominal_cols": nominal_cols,
        "engineered_num_cols": feat.ENGINEERED_NUM_COLS,
        "target": target,
        "y_test_actual": y_test_actual,
        "y_test_pred_actual": y_test_pred_actual,
        "feature_columns": list(X_train.columns),
    }

    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaved trained pipeline + metadata to {MODEL_PATH}")

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
