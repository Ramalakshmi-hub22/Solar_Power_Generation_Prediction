import streamlit as st
import joblib
import pandas as pd


# -----------------------------
# Load trained model and features
# -----------------------------

model = joblib.load("models/solar_power_model.pkl")
features = joblib.load("models/features.pkl")


# -----------------------------
# Streamlit Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Solar Power Generation Prediction",
    page_icon="☀️",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------

st.title("☀️ Solar Power Generation Prediction")

st.write(
    "Predict AC Power Generation using a trained "
    "Random Forest Regression model."
)


# -----------------------------
# User Input Section
# -----------------------------

st.subheader("Enter Solar and Weather Details")


dc_power = st.number_input(
    "DC Power",
    min_value=0.0,
    value=0.0
)

daily_yield = st.number_input(
    "Daily Yield",
    min_value=0.0,
    value=0.0
)

ambient_temperature = st.number_input(
    "Ambient Temperature"
)

module_temperature = st.number_input(
    "Module Temperature"
)

irradiation = st.number_input(
    "Solar Irradiation",
    min_value=0.0,
    value=0.0
)

hour = st.number_input(
    "Hour of Day",
    min_value=0,
    max_value=23,
    value=12,
    step=1
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict AC Power"):

    input_data = pd.DataFrame(
        [[
            dc_power,
            daily_yield,
            ambient_temperature,
            module_temperature,
            irradiation,
            hour
        ]],
        columns=features
    )

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted AC Power: {prediction:.2f}"
    )