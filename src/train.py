"""
Train baseline models (LogReg, RandomForest, XGBoost) on the credit risk dataset.
Mirrors the choices in notebooks/02_baseline_xgb.ipynb.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Allow running as `python src/train.py` by fixing sys.path
try:
    from src.preprocess import (
        CAT_FEATURES,
        NUM_FEATURES,
        build_preprocessor,
        clip_outliers,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback for direct script run
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.preprocess import (
        CAT_FEATURES,
        NUM_FEATURES,
        build_preprocessor,
        clip_outliers,
    )

DATA_PATH = Path("data/raw/credit_risk_dataset.csv")
MODELS_DIR = Path("models")
TARGET_COL = "loan_status"
RANDOM_STATE = 42


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def train_and_eval(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[str, object, List[Dict[str, float]]]:
    """
    Fit multiple models and return the best model name, fitted estimator, and metrics list.
    """
    results: List[Dict[str, float]] = []

    preprocessor = build_preprocessor()

    def fit_eval(model, name: str):
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, proba)
        results.append({"model": name, "val_auc": auc})
        print(f"{name} AUC: {auc:.4f}")
        return model, auc

    # Class balance stats for scale_pos_weight
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos

    models: Dict[str, Pipeline] = {
        "log_reg": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        learning_rate=0.1,
                        max_depth=4,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        scale_pos_weight=scale_pos_weight,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "xgboost_smote": ImbPipeline(
            steps=[
                ("preprocess", preprocessor),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=250,
                        learning_rate=0.1,
                        max_depth=4,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    best_name = None
    best_model = None
    best_auc = -1.0

    for name, model in models.items():
        fitted, auc = fit_eval(model, name)
        if auc > best_auc:
            best_auc = auc
            best_model = fitted
            best_name = name

    return best_name or "", best_model, results


def save_artifacts(model, preprocessor, metrics: List[Dict[str, float]]):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")
    joblib.dump(model, MODELS_DIR / "best_model.pkl")
    with (MODELS_DIR / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved preprocessor and best model to {MODELS_DIR}")


def main():
    df = load_data()
    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df[TARGET_COL]

    # Clip outliers before splitting
    X = clip_outliers(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    best_name, best_model, results = train_and_eval(X_train, y_train, X_val, y_val)
    print("Validation AUCs:", results)
    print(f"Best model: {best_name}")

    # Extract the shared preprocessor from the best pipeline
    preprocessor = best_model.named_steps.get("preprocess")
    save_artifacts(best_model, preprocessor, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline credit risk models.")
    parser.parse_args()  # Placeholder for future CLI args
    main()
