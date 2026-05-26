import streamlit as st
import pickle
import numpy as np
import os

# PAGE CONFIG

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

# CUSTOM BACKGROUND

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #e0f2fe,
        #f0fdf4,
        #fef9c3,
        #ede9fe
    );
}

h1 {
    color: #1e40af !important;
    text-align: center;
}

h2, h3, p {
    color: #1e3a8a !important;
}

label {
    color: #111827 !important;
    font-weight: bold;
}

div[data-baseweb="select"] > div {
    background-color: white !important;
    color: black !important;
    border-radius: 10px;
}

input {
    background-color: white !important;
    color: black !important;
}

.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
}

.stButton>button:hover {
    background-color: #1d4ed8;
    color: white;
}

</style>
""", unsafe_allow_html=True)
# DYNAMIC PATHS

BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "model.pkl"
)

scaler_path = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "scaler.pkl"
)

# LOAD MODEL

model = pickle.load(
    open(model_path, "rb")
)

# LOAD SCALER

scaler = pickle.load(
    open(scaler_path, "rb")
)

# TITLE

st.title("🏦 Loan Approval Prediction")

st.write("Enter Applicant Details")

# USER INPUTS

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

married = st.selectbox(
    "Marital Status",
    ["No", "Yes"]
)

dependents = st.selectbox(
    "Dependents",
    ["0", "1", "2", "3+"]
)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["No", "Yes"]
)

applicant_income = st.number_input(
    "Applicant Income",
    min_value=0,
    value=5000
)

coapplicant_income = st.number_input(
    "Coapplicant Income",
    min_value=0.0,
    value=1500.0
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0.0,
    value=120.0
)

loan_amount_term = st.number_input(
    "Loan Amount Term",
    min_value=0.0,
    value=360.0
)

credit_history = st.selectbox(
    "Credit History",
    [0.0, 1.0]
)

property_area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

# MANUAL ENCODING

gender = 1 if gender == "Male" else 0

married = 1 if married == "Yes" else 0

dependents_map = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3+": 3
}

dependents = dependents_map[dependents]

education = 0 if education == "Graduate" else 1

self_employed = 1 if self_employed == "Yes" else 0

property_map = {
    "Rural": 0,
    "Semiurban": 1,
    "Urban": 2
}

property_area = property_map[property_area]

# PREDICT BUTTON

if st.button("Predict Loan Status"):

    features = np.array([
        [
            gender,
            married,
            dependents,
            education,
            self_employed,
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_amount_term,
            credit_history,
            property_area
        ]
    ])

    # SCALE FEATURES

    scaled_features = scaler.transform(
        features
    )

    # PREDICT

    prediction = model.predict(
        scaled_features
    )

    # DISPLAY RESULT

    if prediction[0] == 1:

        st.success(
            "Loan Approved ✅"
        )

    else:

        st.error(
            "Loan Rejected ❌"
        )
