# Capstone 1: Credit Risk Pipeline

**Dataset** : Kaggle “Credit Risk” (≈325k rows). Columns: age, income, home ownership, employment length, loan intent/grade, amount, interest rate, payment-to-income ratio, prior default flag, credit history length, target `loan_status` (1 = default).    
**Stack** : XGBoost + SMOTE + SHAP + Streamlit/FastAPI + Docker (in progress).

## Problem Statement

**Credit Risk Prediction**: Binary classification of personal loan default (target: loan_status=1). Business impact: Banks lose 5-10% of loan portfolio to defaults; model enables risk-based pricing/denial, reducing expected loss by 20% while approving 80% good loans.

**Model deployment**: XGBoost probability via FastAPI → threshold decisions (>0.5=deny).

## EDA Summary

- Target is imbalanced (~20% defaults) → use ROC-AUC/PR-AUC and class weights or SMOTE.  
- Categorical risk: lower `loan_grade` (C/D/F/G) and prior default flag `cb_person_default_on_file = Y` push default rates up.  
- Numeric risk: higher `loan_int_rate` and higher `loan_percent_income` move together and align with more defaults.  
- Outliers: ages >100, employment length >60, very high rates/payment ratios → clip around age 90, emp_length 60, rate 40, payment ratio 1.5 before scaling.  
- Missingness is low; median/mode imputation is enough.  
- Notebooks: `models/01_eda.ipynb` (EDA) and `models/02_baseline_xgb.ipynb` (baseline modeling).

## Baseline Modeling (notebook 02)

- Features: nums = `person_age`, `person_income`, `person_emp_length`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`, `cb_person_cred_hist_length`; cats = `person_home_ownership`, `loan_intent`, `loan_grade`, `cb_person_default_on_file`.  
- Preprocessing: clip outliers, then ColumnTransformer (median imputer + scaler for nums; mode imputer + OHE for cats). Unknown categories ignored.  
- Models compared on val AUC: LogisticRegression (class_weight balanced), RandomForest (class_weight balanced), XGBoost (scale_pos_weight), optional XGBoost + SMOTE.  
- SHAP cell to peek at feature importance; pick the best model (likely XGB) to carry into scripts.

| Model | Val AUC (notebook) | Val AUC (train.py run) | Notes |
|-------|--------------------|-------------------------|-------|
| LogisticRegression (balanced) | 0.871180 | 0.8712 | Fast baseline, class weights only |
| RandomForest (balanced) | 0.930746 | 0.9307 | 300 trees, min_samples_leaf=2 |
| XGBoost (scale_pos_weight) | **0.949466** | **0.9495** | 300 trees, lr=0.1, depth=4 |
| XGBoost + SMOTE | 0.938520 | 0.9385 | Oversample then XGB |

## Project Status

- ✅ EDA notebooks drafted (`models/01_eda.ipynb`, `models/02_baseline_xgb.ipynb`).  
- ✅ Training/inference scripts implemented (`src/preprocess.py`, `src/train.py`, `src/predict.py`); best model currently XGBoost (val AUC ~0.9495).  
- ✅ FastAPI ready (`uvicorn src.predict:app`) and Dockerfile for serving the API.  
- ✅ Streamlit UI (local) loading saved artifacts.  
- Data: `data/raw/credit_risk_dataset.csv` committed for reproducibility.

## Quickstart

```bash
# 1) Install deps
pip install -r requirements.txt

# 2) Train and save artifacts (preprocessor + best_model.pkl)
python -m src.train

# 3) Serve API locally
uvicorn src.predict:app --reload
```

## Test API

```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" \
-d '{"person_age":35,"person_income":80000,"person_emp_length":5,"loan_amnt":15000,"loan_int_rate":12.5,"loan_percent_income":0.25,"cb_person_cred_hist_length":10,"person_home_ownership":"RENT","loan_intent":"MEDICAL","loan_grade":"C","cb_person_default_on_file":"N"}'
```

Example response: `{"default_probability":0.10,"risk":"LOW"}`

## Docker (API)

Build the image (requires `models/best_model.pkl` and `models/preprocessor.pkl` present):

```bash
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

## Streamlit (local)

Runs locally loading the saved model artifacts (no API call):

```bash
pip install -r requirements.txt
python -m src.train   # ensure models/ are up to date
streamlit run app.py
```

Open `http://localhost:8501`, fill the form, and you’ll get the default probability and HIGH/LOW label. The app also shows top feature importances from the XGBoost model for the current input.

## Useful links

[Kaggle credit risk dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data)  
[DataTalks ML Zoomcamp](https://courses.datatalks.club/ml-zoomcamp-2025/)

## Repo layout (planned)

```
├── README.md
├── data/raw/credit_risk_dataset.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_baseline_xgb.ipynb
├── src/
│   ├── preprocess.py        # build_preprocessor() with clipping + CT
│   ├── train.py             # train Logistic/RF/XGB, save best + preprocessor
│   └── predict.py           # FastAPI /predict using saved artifacts
├── models/
│   ├── best_model.pkl
│   ├── metrics.json
│   └── preprocessor.pkl
├── app.py                   # Streamlit UI (local)
├── Dockerfile               # container for FastAPI
└── requirements.txt
```
