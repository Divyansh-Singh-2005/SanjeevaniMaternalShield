# ===============================
# Sanjeevani Maternal Shield Pipeline
# ===============================

# ---- Risk labels ----
RISK_LABELS = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk"
}

HIGH_RISK_THRESHOLD = 0.60
MEDIUM_RISK_THRESHOLD = 0.40


# ---- Risk categorization ----
def categorize_risk_from_proba(probabilities):
    p_low, p_mid, p_high = probabilities

    if p_high >= HIGH_RISK_THRESHOLD:
        return "High Risk"
    elif p_mid >= MEDIUM_RISK_THRESHOLD:
        return "Medium Risk"
    else:
        return "Low Risk"


# ---- Predict risk ----
def predict_risk_category(model, scaler, input_features_df):
    input_scaled = scaler.transform(input_features_df)
    proba = model.predict_proba(input_scaled)[0]

    risk_category = categorize_risk_from_proba(proba)

    return {
        "risk_category": risk_category,
        "probabilities": {
            "low": round(float(proba[0]), 3),
            "medium": round(float(proba[1]), 3),
            "high": round(float(proba[2]), 3)
        }
    }


# ---- Extract vitals with unit safety ----
def extract_vitals(feature_row):
    bs_raw = float(feature_row["BS"])

    # Convert mmol/L → mg/dL if needed
    if bs_raw < 20:
        bs_value = round(bs_raw * 18, 1)
        bs_unit = "mg/dL (converted from mmol/L)"
    else:
        bs_value = round(bs_raw, 1)
        bs_unit = "mg/dL"

    return {
        "Age": int(feature_row["Age"]),
        "SystolicBP": int(feature_row["SystolicBP"]),
        "DiastolicBP": int(feature_row["DiastolicBP"]),
        "BS": bs_value,
        "BS_unit": bs_unit,
        "BodyTemp": round(float(feature_row["BodyTemp"]), 1),
        "HeartRate": int(feature_row["HeartRate"])
    }


# ---- Hindi medical explanation ----
def english_to_hindi_medical(vitals, risk_output):
    risk = risk_output["risk_category"]

    if risk == "High Risk":
        return f"""
🚨 उच्च जोखिम गर्भावस्था

कारण:
- महत्वपूर्ण संकेत गंभीर जोखिम दर्शाते हैं।
- रक्तचाप: {vitals['SystolicBP']} mmHg
- रक्त शर्करा: {vitals['BS']} {vitals['BS_unit']}

आवश्यक कार्रवाई:
- तुरंत नज़दीकी अस्पताल में रेफर करें।
- देरी न करें।
- हर 1–2 घंटे में BP की निगरानी करें।

नोट:
- उच्च जोखिम की संभावना: {risk_output['probabilities']['high']*100:.1f}%
"""

    elif risk == "Medium Risk":
        return f"""
⚠️ मध्यम जोखिम गर्भावस्था

कारण:
- महत्वपूर्ण संकेत मध्यम जोखिम दर्शाते हैं।
- रक्तचाप: {vitals['SystolicBP']} mmHg
- रक्त शर्करा: {vitals['BS']} {vitals['BS_unit']}

आवश्यक कार्रवाई:
- प्रतिदिन BP की निगरानी करें।
- नमक और शक्कर का सेवन कम करें।
- 24–48 घंटे में डॉक्टर से परामर्श करें।

नोट:
- मध्यम जोखिम की संभावना: {risk_output['probabilities']['medium']*100:.1f}%
"""

    else:
        return f"""
✅ कम जोखिम गर्भावस्था

कारण:
- सभी महत्वपूर्ण संकेत सामान्य सीमा में हैं।
- रक्तचाप: {vitals['SystolicBP']} mmHg
- रक्त शर्करा: {vitals['BS']} {vitals['BS_unit']}

आवश्यक कार्रवाई:
- नियमित जांच जारी रखें।
- संतुलित आहार लें।
- मासिक निगरानी करें।

नोट:
- कम जोखिम की संभावना: {risk_output['probabilities']['low']*100:.1f}%
"""


# ---- End-to-end pipeline ----
def sanjeevani_maternal_shield_pipeline(
    patient_df,
    model,
    scaler,
    speak=False
):
    vitals = extract_vitals(patient_df.iloc[0])

    risk_output = predict_risk_category(
        model=model,
        scaler=scaler,
        input_features_df=patient_df
    )

    hindi_explanation = english_to_hindi_medical(
        vitals=vitals,
        risk_output=risk_output
    )

    return {
        "risk_category": risk_output["risk_category"],
        "probabilities": risk_output["probabilities"],
        "vitals": vitals,
        "hindi_explanation": hindi_explanation
    }
