import streamlit as st
import requests

API_URL = "https://stress-detection-api-my05.onrender.com/predict"

st.set_page_config(page_title="Stress Detection System")

st.title("Stress Detection System")
st.write("Enter the extracted physiological signal features.")

st.header("ECG Features")
ecg_mean = st.number_input("Mean of ECG")
ecg_std = st.number_input("Standard Deviation of ECG")
ecg_min = st.number_input("Minimum of ECG")
ecg_max = st.number_input("Maximum of ECG")
ecg_median = st.number_input("Median of ECG")

st.header("EDA Features")
eda_mean = st.number_input("Mean of EDA")
eda_std = st.number_input("Standard Deviation of EDA")
eda_min = st.number_input("Minimum of EDA")
eda_max = st.number_input("Maximum of EDA")
eda_median = st.number_input("Median of EDA")

st.header("Respiration Features")
resp_mean = st.number_input("Mean of Respiration")
resp_std = st.number_input("Standard Deviation of Respiration")
resp_min = st.number_input("Minimum of Respiration")
resp_max = st.number_input("Maximum of Respiration")
resp_median = st.number_input("Median of Respiration")

st.header("Temperature Features")
temp_mean = st.number_input("Mean of Temperature")
temp_std = st.number_input("Standard Deviation of Temperature")
temp_min = st.number_input("Minimum of Temperature")
temp_max = st.number_input("Maximum of Temperature")
temp_median = st.number_input("Median of Temperature")

if st.button("Predict Stress Level"):

    features = [
        ecg_mean, ecg_std, ecg_min, ecg_max, ecg_median,
        eda_mean, eda_std, eda_min, eda_max, eda_median,
        resp_mean, resp_std, resp_min, resp_max, resp_median,
        temp_mean, temp_std, temp_min, temp_max, temp_median
    ]

    payload = {"features": features}

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()

            st.success(f"Predicted State: {result['label']}")

            st.subheader("Class Probabilities")
            st.json(result["probabilities"])

        else:
            st.error(f"API Error: {response.text}")

    except Exception as e:
        st.error(f"Connection Error: {e}")