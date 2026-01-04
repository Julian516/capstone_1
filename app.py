import joblib
import pandas as pd
import streamlit as st
import shap
from pathlib import Path

# Allow running as `streamlit run app.py` from project root
try:
    from src.preprocess import CAT_FEATURES, NUM_FEATURES, clip_outliers
except ModuleNotFoundError:  # pragma: no cover - fallback for direct script run
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from src.preprocess import CAT_FEATURES, NUM_FEATURES, clip_outliers

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """
    Load model (pipeline) and preprocessor. The model already contains the
    preprocessor when trained via src/train.py, so we keep the standalone
    preprocessor as a fallback.
    """
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    return model, preprocessor


def main():
    st.set_page_config(page_title="Credit Risk Scoring", layout="centered")
    st.title("Credit Risk Scoring")
    st.caption("Local Streamlit app using saved preprocessor + XGBoost model")

    model, preprocessor = load_artifacts()

    with st.form("loan_form"):
        st.subheader("Borrower information")
        col1, col2 = st.columns(2)
        with col1:
            person_age = st.number_input("Age", min_value=18, max_value=100, value=35)
            person_income = st.number_input("Annual income ($)", min_value=0, max_value=500000, value=80000)
            person_emp_length = st.number_input("Employment length (years)", min_value=0.0, max_value=60.0, value=5.0, step=0.5)
            cb_person_cred_hist_length = st.number_input("Credit history length (years)", min_value=0, max_value=50, value=10)
        with col2:
            person_home_ownership = st.selectbox("Home ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"], index=0)
            loan_intent = st.selectbox("Loan intent", ["MEDICAL", "EDUCATION", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT", "PERSONAL", "VENTURE"], index=0)
            loan_grade = st.selectbox("Loan grade", ["A", "B", "C", "D", "E", "F", "G"], index=2)
            cb_person_default_on_file = st.selectbox("Prior default on file", ["N", "Y"], index=0)

        st.subheader("Loan details")
        loan_amnt = st.number_input("Loan amount ($)", min_value=500, max_value=100000, value=15000, step=500)
        loan_int_rate = st.number_input("Interest rate (%)", min_value=0.0, max_value=60.0, value=12.5, step=0.1)
        loan_percent_income = st.number_input("Payment-to-income ratio", min_value=0.0, max_value=5.0, value=0.25, step=0.01)

        submitted = st.form_submit_button("Predict risk")

    if submitted:
        payload = {
            "person_age": float(person_age),
            "person_income": float(person_income),
            "person_emp_length": float(person_emp_length),
            "loan_amnt": float(loan_amnt),
            "loan_int_rate": float(loan_int_rate),
            "loan_percent_income": float(loan_percent_income),
            "cb_person_cred_hist_length": float(cb_person_cred_hist_length),
            "person_home_ownership": person_home_ownership,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "cb_person_default_on_file": cb_person_default_on_file,
        }

        df = pd.DataFrame([payload])[NUM_FEATURES + CAT_FEATURES]
        df = clip_outliers(df)

        # If the model is a pipeline with preprocessing inside, use it directly.
        if hasattr(model, "named_steps") and "preprocess" in model.named_steps:
            proba = model.predict_proba(df)[0, 1]
        else:
            # Fallback: apply standalone preprocessor then predict
            X = preprocessor.transform(df)
            proba = model.predict_proba(X)[0, 1]
        risk = "HIGH" if proba >= 0.5 else "LOW"

        st.metric(label="Default probability", value=f"{proba:.2%}")
        st.success(f"Risk: {risk}")

        # Simple feature importance bar chart (XGBoost feature_importances_)
        try:
            if hasattr(model, "named_steps") and "preprocess" in model.named_steps:
                prep = model.named_steps["preprocess"]
                est = model.named_steps.get("model", model)
                feature_names = prep.get_feature_names_out()
            else:
                prep = preprocessor
                est = model
                feature_names = prep.get_feature_names_out()

            if hasattr(est, "feature_importances_"):
                importances = est.feature_importances_
                contrib = (
                    pd.DataFrame({"feature": feature_names, "importance": importances})
                    .assign(abs_val=lambda d: d["importance"].abs())
                    .sort_values("abs_val", ascending=False)
                    .head(10)
                )
                st.subheader("Top feature importance (model)")
                st.bar_chart(contrib.set_index("feature")["importance"])
            else:
                st.caption("Feature importances not available for this model type.")
        except Exception as e:
            st.caption(f"Feature importance unavailable: {e}")


if __name__ == "__main__":
    main()
