import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
heart_dataset = pd.read_csv("D:/Z/projects/heart disease/heart_disease_data.csv")

# Split into features & target
X = heart_dataset.drop(columns='target', axis=1)
Y = heart_dataset['target']

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, Y_train)

# Streamlit App
st.set_page_config(page_title="❤️ Heart Disease Prediction", page_icon="🩺", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details below to check the risk of heart disease.")

# Sidebar - Model Performance
st.sidebar.header("📊 Model Performance")
train_acc = accuracy_score(model.predict(X_train), Y_train)
test_acc = accuracy_score(model.predict(X_test), Y_test)
st.sidebar.write(f"✅ Training Accuracy: **{train_acc:.2f}**")
st.sidebar.write(f"✅ Testing Accuracy: **{test_acc:.2f}**")

# Input form
with st.form("heart_form"):
    age = st.number_input("Age", min_value=20, max_value=100, value=50)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [0, 1])
    restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, format="%.1f")
    slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
    ca = st.selectbox("Major Vessels Colored (0-4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", [1, 2, 3])

    submitted = st.form_submit_button("🔍 Predict")

# Prediction result
if submitted:
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)

    if prediction[0] == 0:
        st.success("✅ The person does NOT have heart disease.")
    else:
        st.error("⚠️ The person DOES have heart disease.")
