"""
FastAPI service for credit risk prediction.
Loads the trained preprocessor + best model saved by train.py.
"""

from __future__ import annotations

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

# Allow running as `python src/predict.py` by fixing sys.path
try:
    from src.preprocess import CAT_FEATURES, NUM_FEATURES, clip_outliers
except ModuleNotFoundError:  # pragma: no cover - fallback for direct script run
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.preprocess import CAT_FEATURES, NUM_FEATURES, clip_outliers

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
THRESHOLD = 0.5

app = FastAPI(title="Credit Risk API", version="0.1.0")

# Load artifacts at startup
preprocessor = joblib.load(PREPROCESSOR_PATH)
model = joblib.load(MODEL_PATH)


class LoanData(BaseModel):
    person_age: float
    person_income: float
    person_emp_length: float
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_cred_hist_length: float
    person_home_ownership: str
    loan_intent: str
    loan_grade: str
    cb_person_default_on_file: str


@app.post("/predict")
def predict(data: LoanData):
    # Convert incoming payload to DataFrame with expected column order
    df = pd.DataFrame([data.dict()])[NUM_FEATURES + CAT_FEATURES]

    # Apply same clipping as training
    df = clip_outliers(df)

    proba = model.predict_proba(df)[0, 1]
    risk = "HIGH" if proba >= THRESHOLD else "LOW"
    return {"default_probability": float(proba), "risk": risk}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.predict:app", host="0.0.0.0", port=8000, reload=True)
