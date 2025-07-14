import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('heart_disease_model.pkl')

# App title and subtitle
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

st.markdown("<h1 style='text-align: center;'>❤️ Heart Disease Risk Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Enter the patient's medical data below to predict the likelihood of heart disease.</h4>", unsafe_allow_html=True)
st.write("")

# Split inputs into 2 columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ["Female", "Male"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina (0)", "Atypical Angina (1)", "Non-anginal (2)", "Asymptomatic (3)"])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 400, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120?", ["No (0)", "Yes (1)"])

with col2:
    restecg = st.selectbox("Resting ECG", ["Normal (0)", "ST-T Abnormality (1)", "Left Ventricular Hypertrophy (2)"])
    thalach = st.number_input("Max Heart Rate Achieved", 70, 210, 150)
    exang = st.selectbox("Exercise Induced Angina", ["No (0)", "Yes (1)"])
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST", ["Upsloping (0)", "Flat (1)", "Downsloping (2)"])
    ca = st.selectbox("Major Vessels Colored (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", ["Normal (1)", "Fixed Defect (2)", "Reversible Defect (3)"])

# Map inputs back to model-friendly values
sex = 1 if sex == "Male" else 0
cp = int(cp.split("(")[1][0])
fbs = 1 if fbs == "Yes (1)" else 0
restecg = int(restecg.split("(")[1][0])
exang = 1 if exang == "Yes (1)" else 0
slope = int(slope.split("(")[1][0])
thal = int(thal.split("(")[1][0])

# Prediction section
if st.button("🔍 Predict Risk"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]  # Probability of class 1

    st.write("---")
    if prediction == 1:
        st.error(f"⚠️ High Risk: The model predicts a {round(proba*100)}% chance of heart disease.")
    else:
        st.success(f"✅ Low Risk: The model predicts a {round((1-proba)*100)}% chance of no heart disease.")
    st.write("---")
    st.markdown("*Note: This is a data-driven prediction and should not replace medical advice.*")