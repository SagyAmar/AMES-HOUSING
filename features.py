"""
features.py
------------
Single source of truth for every column-list / feature-engineering rule taken
directly from the original Ames Housing notebook. Both train.py and app.py
import from here so that the *exact* preprocessing used during training is
guaranteed to be the one applied again at prediction time.

Nothing in this file was invented: every column name, category order and rule
below is copied from the notebook's own "Feature Engineering" / "Data
Preprocessing" sections.
"""

import numpy as np

RANDOM_STATE = 42
TARGET = "SalePrice"

# ---------------------------------------------------------------------------
# 1. Column groups (from the notebook's "Identifying feature types" section)
# ---------------------------------------------------------------------------

# Ordinal features -> explicit rank order (low -> high), taken directly from
# data_description.txt (notebook Section 1 / "Feature Engineering").
ORDINAL_FEATURES = {
    "LotShape":     ["IR3", "IR2", "IR1", "Reg"],
    "LandSlope":    ["Sev", "Mod", "Gtl"],
    "ExterQual":    ["Po", "Fa", "TA", "Gd", "Ex"],
    "ExterCond":    ["Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtQual":     ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtCond":     ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtExposure": ["NA", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "HeatingQC":    ["Po", "Fa", "TA", "Gd", "Ex"],
    "KitchenQual":  ["Po", "Fa", "TA", "Gd", "Ex"],
    "Functional":   ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    "FireplaceQu":  ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageFinish": ["NA", "Unf", "RFn", "Fin"],
    "GarageQual":   ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageCond":   ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "PavedDrive":   ["N", "P", "Y"],
    "PoolQC":       ["NA", "Fa", "TA", "Gd", "Ex"],
    "Fence":        ["NA", "MnWw", "GdWo", "MnPrv", "GdPrv"],
    "Utilities":    ["ELO", "NoSeWa", "NoSewr", "AllPub"],
}
ORDINAL_COLS = list(ORDINAL_FEATURES.keys())

NOMINAL_COLS = [
    "MSSubClass", "MSZoning", "Street", "Alley", "LandContour", "LotConfig",
    "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle",
    "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType",
    "Foundation", "Heating", "CentralAir", "Electrical", "GarageType",
    "MiscFeature", "SaleType", "SaleCondition", "MoSold",
]

# Columns where NaN literally means "feature not present" -> filled with the
# sentinel "NA" (already a valid category level in the data dictionary),
# BEFORE assigning ordinal encodings.
NONE_MEANS_MISSING_COLS = [
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence",
    "Alley", "MiscFeature", "GarageType", "MasVnrType",
]

# Engineered numerical features added in Section 4 of the notebook.
ENGINEERED_NUM_COLS = ["TotalSF", "HouseAge", "RemodAge", "TotalBaths"]

# Count/rating/year-type numerical columns excluded from the automatic
# skew-correction step (a log transform doesn't make sense for these).
EXCLUDE_FROM_SKEW_FIX = [
    "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "GarageYrBlt",
    "YrSold", "MoSold", "HouseAge", "RemodAge", "FullBath", "HalfBath",
    "BsmtFullBath", "BsmtHalfBath", "BedroomAbvGr", "KitchenAbvGr",
    "TotRmsAbvGrd", "Fireplaces", "GarageCars", "TotalBaths",
]

SKEW_THRESHOLD = 0.75


def normalize_columns(df):
    """Remove spaces / slashes from raw AmesHousing.csv column headers so
    they behave as valid identifiers, e.g. "Lot Area" -> "LotArea",
    "1st Flr SF" -> "1stFlrSF". Mirrors the notebook's Section 1 step."""
    df = df.copy()
    df.columns = [c.replace(" ", "").replace("/", "") for c in df.columns]
    return df


def get_numerical_cols(df):
    """Numerical columns = every column that isn't the target, an ordinal
    column, or a nominal column (same rule the notebook uses)."""
    exclude = set(ORDINAL_COLS) | set(NOMINAL_COLS) | {TARGET}
    return [c for c in df.columns if c not in exclude]


def fill_none_means_missing(df):
    """Fill the 'NA-means-no-such-feature' columns with the sentinel 'NA'."""
    df = df.copy()
    for c in NONE_MEANS_MISSING_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("NA")
    return df


def add_engineered_features(df):
    """Row-wise engineered features from the notebook's Section 4:
    TotalSF, HouseAge, RemodAge, TotalBaths."""
    df = df.copy()
    df["TotalSF"] = df["TotalBsmtSF"].fillna(0) + df["1stFlrSF"].fillna(0) + df["2ndFlrSF"].fillna(0)
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["TotalBaths"] = (
        df["FullBath"].fillna(0)
        + 0.5 * df["HalfBath"].fillna(0)
        + df["BsmtFullBath"].fillna(0)
        + 0.5 * df["BsmtHalfBath"].fillna(0)
    )
    return df


def prepare_base_features(raw_df):
    """The two deterministic, row-wise steps that the notebook applies to the
    FULL dataset before ever splitting: NA-sentinel fill + engineered
    columns. Safe to reuse identically on a single prediction row."""
    df = fill_none_means_missing(raw_df)
    df = add_engineered_features(df)
    return df


def get_skew_candidates(num_cols):
    return [c for c in num_cols if c not in EXCLUDE_FROM_SKEW_FIX]


def compute_skewed_cols(X_train, num_cols):
    """Compute which numerical columns are skewed enough (|skew| > 0.75) to
    log1p-transform, using the TRAIN split only (no leakage)."""
    skew_candidates = get_skew_candidates(num_cols)
    skewness = X_train[skew_candidates].apply(lambda s: s.skew())
    return skewness[skewness.abs() > SKEW_THRESHOLD].index.tolist()


def apply_skew_transform(df, skewed_cols):
    """Apply the already-decided log1p transform to the given columns."""
    df = df.copy()
    for c in skewed_cols:
        df[c] = np.log1p(df[c].clip(lower=0))
    return df
