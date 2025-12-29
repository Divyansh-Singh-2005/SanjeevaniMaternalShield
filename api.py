from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI(title="Sanjeevani Maternal Shield API")

model = joblib.load("MaternalHealthRiskDataSet.csv")
scaler = joblib.load("scaler.pkl")

from sanjeevani_pipeline import sanjeevani_maternal_shield_pipeline

@app.post("/predict")
def predict(payload: dict):
    df = pd.DataFrame([payload])
    result = sanjeevani_maternal_shield_pipeline(
        patient_df=df,
        model=model,
        scaler=scaler,
        speak=False
    )
    return result
