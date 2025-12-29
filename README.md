# SanjeevaniMaternalShield
Sanjeevani Maternal Shield is an AI-based system that predicts maternal risk using clinical vitals and delivers Hindi voice guidance for ASHA workers, enabling timely decisions in rural healthcare settings.


Markdown

# 🩺 Sanjeevani Maternal Shield

**Sanjeevani Maternal Shield** is an AI-powered maternal health decision-support system designed to reduce maternal mortality in rural and low-resource settings. It predicts pregnancy risk using basic clinical vitals and delivers clear, Hindi voice-based clinical guidance for frontline ASHA workers when doctors are not immediately available.

Aligned with **UN SDG 3.1**, the system is low-cost, explainable, offline-capable, and deployable in real-world healthcare environments.

---

## 🚨 Problem Statement
Maternal mortality remains high in rural regions due to:
* Delayed risk identification
* Limited access to doctors
* Overburdened frontline health workers
* Lack of actionable, easy-to-understand guidance

**Existing systems focus on data collection, not decision support.**

---

## 💡 Solution Overview
Sanjeevani Maternal Shield bridges this gap by combining **Machine Learning + GenAI-style explanations** to support on-the-ground clinical decisions.

**What it does:**
* Uses basic vitals (BP, blood sugar, heart rate, etc.)
* Predicts **Low / Medium / High** pregnancy risk
* Converts predictions into clinically safe, Hindi explanations
* Provides **offline voice guidance** for ASHA workers
* Works even in low-connectivity environments

---

## 🧠 System Architecture

```mermaid
graph TD
    A[Patient Vitals] --> B[XGBoost Risk Prediction Model]
    B --> C[Probability-Based Risk Logic]
    C --> D[Clinical Explanation Engine]
    D --> E[Hindi Text + Offline Voice]
    E --> F[Frontline Health Worker Action]
(If the diagram above does not render, the flow is: Patient Vitals → XGBoost Model → Risk Logic → Explanation Engine → Hindi Voice → ASHA Worker Action)

🔬 Model & Methodology
Model: XGBoost (multi-class classification)

Evaluation: Stratified 5-Fold Cross Validation

Mean CV Accuracy: ~80.5%

Focus: High-risk recall prioritized for clinical safety

Input Features:

Age

Systolic BP

Diastolic BP

Blood Sugar (unit-safe handling)

Body Temperature

Heart Rate

Risk Categories:

Low Risk

Medium Risk

High Risk

⚠️ Note: In healthcare, reducing false negatives for high-risk cases is prioritized over raw accuracy.

🧬 GenAI Explanation Layer (Core Innovation)
Instead of raw predictions, the system generates:

Human-readable clinical explanations

Rule-based, doctor-aligned recommendations

Deterministic outputs (no hallucinated advice)

This ensures: Transparency, Trust, Safety, and Regulatory friendliness.

🌍 Multilingual & Offline Support
Hindi clinical explanations

Offline text-to-speech (no internet required)

Graceful fallback if regional voices are unavailable

Designed specifically for ASHA workers

🖥 Deployment Available Interfaces
Streamlit App (mobile & tablet friendly)

FastAPI Backend (scalable deployment)

Offline-capable local execution

Run Locally
Bash

pip install -r requirements.txt
streamlit run app.py
🛠 Tech Stack
Python

XGBoost

Scikit-learn

Pandas / NumPy

Streamlit

pyttsx3 (offline voice)

Joblib

📈 Impact
Early identification of maternal risk

Reduced dependency on immediate doctor availability

Actionable guidance at the point of care

Scalable to rural and low-resource settings

Directly aligned with UN SDG 3.1 – Reducing Maternal Mortality

👤 Author
Divyansh Singh Aspiring AI/ML Engineer | Focused on building real-world, impact-driven AI systems

📌 Note for Reviewers
This project prioritizes clinical safety, interpretability, and deployability over leaderboard-style optimization, reflecting real-world healthcare AI constraints.
