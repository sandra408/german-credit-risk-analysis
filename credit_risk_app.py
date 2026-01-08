import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ------------------ LOAD ARTIFACTS ------------------
model = pickle.load(open("credit_risk_app.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
pca = pickle.load(open("pca.pkl", "rb"))

st.title("German Credit Risk Prediction App")
st.write("Enter customer details to predict Good / Bad Credit Risk")

# ------------------ USER INPUT ------------------
Age = st.number_input("Age", min_value=18, max_value=100, value=30)

Sex = st.selectbox("Sex", ["male", "female"])
Housing = st.selectbox("Housing", ["own", "rent", "free"])
Saving = st.selectbox("Saving accounts", ["little", "moderate", "rich", "quite rich", "Unknown"])
Checking = st.selectbox("Checking account", ["little", "moderate", "rich", "Unknown"])

Job = st.selectbox("Job", [0, 1, 2, 3])
CreditAmount = st.number_input("Credit Amount", min_value=0, value=1000)
Duration = st.number_input("Duration (months)", min_value=1, value=12)

Purpose = st.selectbox(
    "Purpose",
    [
        "business",
        "car",
        "domestic appliances",
        "education",
        "furniture/equipment",
        "radio/TV",
        "repairs",
        "vacation/others",
    ],
)

# ------------------ LABEL ENCODING ------------------
label_maps = {
    "Sex": {"male": 1, "female": 0},
    "Housing": {"own": 2, "rent": 1, "free": 0},
    "Saving accounts": {"little": 0, "moderate": 1, "rich": 2, "quite rich": 3, "Unknown": 4},
    "Checking account": {"little": 0, "moderate": 1, "rich": 2, "Unknown": 3},
}

Sex = label_maps["Sex"][Sex]
Housing = label_maps["Housing"][Housing]
Saving = label_maps["Saving accounts"][Saving]
Checking = label_maps["Checking account"][Checking]

# ------------------ ONE HOT ENCODING (PURPOSE) ------------------
purpose_cols = [
    "business",
    "car",
    "domestic appliances",
    "education",
    "furniture/equipment",
    "radio/TV",
    "repairs",
    "vacation/others",
]

purpose_data = {col: 0 for col in purpose_cols}
purpose_data[Purpose] = 1

# ------------------ BUILD INPUT DATAFRAME ------------------
input_data = pd.DataFrame(
    [
        {
            "Age": Age,
            "Sex": Sex,
            "Job": Job,
            "Housing": Housing,
            "Saving accounts": Saving,
            "Checking account": Checking,
            "Credit amount": CreditAmount,
            "Duration": Duration,
            **purpose_data,
        }
    ]
)

# ------------------ FORCE TRAINING FEATURE ORDER ------------------
input_data = input_data[scaler.feature_names_in_]

# ------------------ PREDICTION ------------------
if st.button("Predict"):
    scaled = scaler.transform(input_data)
    pca_features = pca.transform(scaled)
    prediction = model.predict(pca_features)[0]

    if prediction == 1:
        st.success("GOOD Customer (Low Credit Risk)")
    else:
        st.error(" BAD Customer (High Credit Risk)")
