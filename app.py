import streamlit as st
import pandas as pd
import joblib

# ===============================
# Load trained ML artifacts
# ===============================
xgb = joblib.load("maternal_risk_model.pkl")
scaler = joblib.load("scaler.pkl")

# ===============================
# Import pipeline
# ===============================
from sanjeevani_pipeline import sanjeevani_maternal_shield_pipeline
# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Sanjeevani Maternal Shield", layout="centered")

st.title("🩺 Sanjeevani Maternal Shield")
st.caption("AI-powered maternal risk assessment for frontline health workers")

age = st.number_input("Age", 15, 50, 25)
sbp = st.number_input("Systolic BP (mmHg)", 80, 200, 120)
dbp = st.number_input("Diastolic BP (mmHg)", 50, 130, 80)
bs = st.number_input("Blood Sugar", 60.0, 300.0, 120.0)
temp = st.number_input("Body Temperature (°F)", 95.0, 104.0, 98.6)
hr = st.number_input("Heart Rate", 50, 140, 80)

if st.button("Assess Risk"):
    patient_df = pd.DataFrame([{
        "Age": age,
        "SystolicBP": sbp,
        "DiastolicBP": dbp,
        "BS": bs,
        "BodyTemp": temp,
        "HeartRate": hr
    }])

    output = sanjeevani_maternal_shield_pipeline(
        patient_df=patient_df,
        model=xgb,
        scaler=scaler,
        speak=False
    )

    st.subheader("Risk Category")
    st.success(output["risk_category"])

    st.subheader("Clinical Guidance (Hindi)")
    st.text(output["hindi_explanation"])
