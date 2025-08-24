# ❤️ Heart Disease Prediction – Logistic Regression

A machine learning project that predicts the presence of heart disease based on patient data, using the Logistic Regression model. Designed with both training/evaluation pipelines and a live Streamlit app for user interaction.

## 🌐 Live Demo

Try the Streamlit app here: [Heart Disease Predictor](https://heart-disease-7m2oe79xb4mjgzx6qvbqlu.streamlit.app/)

## 📌 Project Overview

Objective: Use diagnostic/patient data to predict heart disease (binary classification).

Model: Logistic Regression — simple, interpretable, and effective for binary outcomes.

Includes:

Data loading & preprocessing

Feature selection and engineering

Model training and evaluation (accuracy, ROC AUC)

Streamlit-powered UI (app.py)

## 📂 Repository Structure

/Heart_Disease_Prediction_Logistic_Regression

│── heart_disease_data.csv     # Dataset

│── app.py                     # Streamlit app interface

│── requirements.txt           # Python dependencies

│── README.md                  # Documentation

## ⚡ Getting Started

### 1⃣ Clone the Repository

git clone https://github.com/abhinav744/Heart_Disease_Prediction_Logistic_Regression.git

cd Heart_Disease_Prediction_Logistic_Regression

### 2⃣ (Optional) Set Up a Virtual Environment

python -m venv venv

source venv/bin/activate       # On macOS/Linux

venv\Scripts\activate          # On Windows

### 3⃣ Install Dependencies

pip install -r requirements.txt

### 4⃣ Run the Streamlit App Locally

streamlit run app.py

## 📊 Insights & Typical Performance

Accuracy: 80–85%

ROC AUC: 0.80–0.90

Key predictors: cholesterol, age, blood pressure, max heart rate

## 🚀 Future Enhancements

Add regularization or hyperparameter tuning (GridSearchCV)

Compare with advanced models (Random Forest, XGBoost, Neural Networks)

Visual dashboards (ROC curve, feature importance)
