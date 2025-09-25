import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model artifacts
model = joblib.load("archive/logistic_regression_model.pkl")
scaler = joblib.load("archive/minmax_scaler.pkl")
features = joblib.load("archive/features.pkl")

st.title("Crash Severity Prediction")

st.markdown("""
This app predicts whether a crash will be **severe (1)** or **not severe (0)** 
based on selected features. Adjust the sidebar and click **Predict**.
""")

# Collect user inputs
user_input = {}
st.sidebar.header("Input Features")
for feat in features:
    if feat in ['crash_hour', 'HOUR']:
        user_input[feat] = st.sidebar.slider(f"{feat} (0-23)", 0, 23, 12)
    elif feat == 'crash_month':
        user_input[feat] = st.sidebar.slider(f"{feat} (1-12)", 1, 12, 6)
    elif feat == 'MINUTE':
        user_input[feat] = st.sidebar.slider(f"{feat} (0-59)", 0, 59, 0)
    elif feat == 'is_night':
        user_input[feat] = st.sidebar.selectbox(f"{feat}", [0, 1])
    else:
        user_input[feat] = st.sidebar.number_input(feat, 0, 1000, 0)

input_df = pd.DataFrame([user_input])

st.subheader("Input Features")
st.write(input_df)

# Predict button
if st.button("Predict"):
    # Scale input
    input_scaled = scaler.transform(input_df)
    # Predict class
    pred_class = model.predict(input_scaled)[0]
    # Predict probabilities
    pred_proba = model.predict_proba(input_scaled)[0]
    
    st.subheader("Prediction Results")
    st.write(f"Predicted Severity Class: **{pred_class}**")
    st.write("Class Probabilities:")
    st.write({
        "Not Severe (0)": f"{pred_proba[0]*100:.2f}%",
        "Severe (1)": f"{pred_proba[1]*100:.2f}%"
    })
else:
    st.info("Adjust input features in the sidebar and click **Predict** to see results.")
