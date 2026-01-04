"""
Preprocessing utilities: feature lists, outlier clipping, and a ColumnTransformer
that matches the baseline notebook choices.
"""

from typing import Iterable, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Feature lists aligned with notebooks/02_baseline_xgb.ipynb
NUM_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

CAT_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

# Simple clipping thresholds to limit extreme values before scaling
CLIP_LIMITS = {
    "person_age": (None, 90),
    "person_emp_length": (None, 60),
    "loan_int_rate": (None, 40),
    "loan_percent_income": (None, 1.5),
}


def clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with selected numeric columns clipped to reasonable bounds.
    """
    clipped = df.copy()
    for col, (lower, upper) in CLIP_LIMITS.items():
        if col in clipped.columns:
            clipped[col] = clipped[col].clip(lower=lower, upper=upper)
    return clipped


def build_preprocessor(
    num_features: Iterable[str] = NUM_FEATURES,
    cat_features: Iterable[str] = CAT_FEATURES,
) -> ColumnTransformer:
    """
    Build the ColumnTransformer used for model training:
    - Numeric: median imputer + standard scaler
    - Categorical: most-frequent imputer + one-hot encoder (ignore unknowns)
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(num_features)),
            ("cat", categorical_transformer, list(cat_features)),
        ]
    )
    return preprocessor


__all__: Tuple[str, ...] = (
    "NUM_FEATURES",
    "CAT_FEATURES",
    "CLIP_LIMITS",
    "clip_outliers",
    "build_preprocessor",
)
