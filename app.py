import streamlit as st
import pandas as pd
import joblib 

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="wide"
)

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #020617 0%, #0f172a 50%, #164e63 100%);
}

.section-title {
    color: #67e8f9;
}

p, label, .stMarkdown {
    color: #e2e8f0;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Header */
.hero {
    background: linear-gradient(135deg, #0f766e, #0891b2);
    padding: 35px;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.10);
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 10px;
}

.hero p {
    font-size: 18px;
    opacity: 0.9;
}

/* Section headings */
.section-title {
    font-size: 24px;
    font-weight: 700;
    color: #164e63;
    margin-top: 25px;
    margin-bottom: 15px;
}

/* Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #0f766e, #0891b2);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 15px;
    font-size: 18px;
    font-weight: 700;
}

.stButton > button:hover {
    box-shadow: 0 8px 20px rgba(8,145,178,0.3);
    transform: translateY(-2px);
}

/* Result cards */
.result-card {
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    margin-top: 25px;
}

.low-risk {
    background: #ecfdf5;
    border: 2px solid #10b981;
    color: #065f46;
}

.high-risk {
    background: #fef2f2;
    border: 2px solid #ef4444;
    color: #991b1b;
}

</style>
""", unsafe_allow_html=True)

model=joblib.load('Logistic_Regression_Of_Heart.pkl')
scaler=joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

st.markdown("""
<div class="hero">
    <h1>🫀 Heart Disease Risk Assessment</h1>
    <p>
        Enter the patient's health information below
        to estimate cardiovascular risk.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="section-title">👤 Patient Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    age = st.slider('Age', 18, 100, 50)

with col2:
    sex = st.selectbox("Sex", ['Male', 'Female'])


chest_pain_type=st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])


resting_bp=st.number_input("Resting Blood Pressure (in mm Hg)", 80, 200, 120)


cholesterol=st.number_input("Cholesterol (in mg/dl)", 100, 600, 200)


fasting_bs=st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])


resting_ecg=st.selectbox("Resting Electrocardiographic Results", ["Normal", "ST", "LVH"])


max_hr=st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)


exercise_angina=st.selectbox("Exercise Induced Angina", ["Yes", "No"])


oldpeak=st.number_input("Oldpeak (ST depression induced by exercise relative to rest)", 0.0, 6.0, 1.0)

st_slope=st.selectbox("Slope of the peak exercise ST segment", ["Up", "Flat", "Down"])


if st.button("Predict"):
    input_df = pd.DataFrame({
        'age': [age],
        'sex': [1 if sex == 'Male' else 0],
        'cp': [chest_pain_type],                                                
        'trestbps': [resting_bp],
        'chol': [cholesterol],  
        'fbs': [1 if fasting_bs == 'Yes' else 0],
        'restecg': [resting_ecg],
        'thalach': [max_hr],
        'exang': [1 if exercise_angina == 'Yes' else 0],
        'oldpeak': [oldpeak],
        'slope': [st_slope]
    }) 

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("The model predicts that you have a high risk of heart disease.Please consult a healthcare professional for further evaluation.")        
    else:
        st.success("The model predicts that you have a low risk of heart disease. However, it is always advisable to maintain a healthy lifestyle and consult a healthcare professional for regular check-ups.")