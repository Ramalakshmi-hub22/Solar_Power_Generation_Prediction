import streamlit as st
import joblib
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Solar Power Generation Prediction",
    page_icon="☀️",
    layout="wide"
)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "solar_power_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error("❌ Model file not found!")
    st.write("Expected model location:")
    st.code(str(MODEL_PATH))

    st.info(
        "Make sure your GitHub repository contains "
        "`models/solar_power_model.pkl`."
    )

    st.stop()

except Exception as e:
    st.error("❌ Error loading the trained model.")
    st.exception(e)
    st.stop()


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------
st.title("☀️ Solar Power Generation Prediction")
st.markdown(
    "Predict solar power generation using environmental and "
    "weather-related parameters."
)

st.divider()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("🌤️ Input Parameters")

st.sidebar.subheader("Weather Conditions")

temperature = st.sidebar.number_input(
    "Temperature (°C)",
    min_value=-20.0,
    max_value=60.0,
    value=25.0,
    step=0.1
)

humidity = st.sidebar.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=60.0,
    step=0.1
)

wind_speed = st.sidebar.number_input(
    "Wind Speed (m/s)",
    min_value=0.0,
    max_value=100.0,
    value=5.0,
    step=0.1
)

solar_irradiance = st.sidebar.number_input(
    "Solar Irradiance (W/m²)",
    min_value=0.0,
    max_value=1500.0,
    value=500.0,
    step=1.0
)


# ---------------------------------------------------------
# ADDITIONAL INPUTS
# ---------------------------------------------------------
st.sidebar.subheader("☀️ Solar Conditions")

panel_temperature = st.sidebar.number_input(
    "Panel Temperature (°C)",
    min_value=-20.0,
    max_value=100.0,
    value=30.0,
    step=0.1
)

cloud_cover = st.sidebar.number_input(
    "Cloud Cover (%)",
    min_value=0.0,
    max_value=100.0,
    value=20.0,
    step=1.0
)


# ---------------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------------
predict_button = st.sidebar.button(
    "🔮 Predict Solar Power",
    use_container_width=True
)


# ---------------------------------------------------------
# MAIN CONTENT
# ---------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "🌡️ Temperature",
        f"{temperature:.1f} °C"
    )

with col2:
    st.metric(
        "💧 Humidity",
        f"{humidity:.1f} %"
    )

with col3:
    st.metric(
        "☀️ Irradiance",
        f"{solar_irradiance:.0f} W/m²"
    )


st.divider()


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
if predict_button:

    # Create input dataframe
    input_data = pd.DataFrame({
        "temperature": [temperature],
        "humidity": [humidity],
        "wind_speed": [wind_speed],
        "solar_irradiance": [solar_irradiance],
        "panel_temperature": [panel_temperature],
        "cloud_cover": [cloud_cover]
    })

    try:

        prediction = model.predict(input_data)

        predicted_value = float(prediction[0])

        st.success("✅ Prediction completed successfully!")

        st.subheader("🔋 Predicted Solar Power Generation")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "Predicted Power",
                f"{predicted_value:.2f}"
            )

        with result_col2:
            st.metric(
                "Solar Irradiance",
                f"{solar_irradiance:.0f} W/m²"
            )

        st.divider()

        st.subheader("📊 Input Parameters")

        display_data = pd.DataFrame({
            "Parameter": [
                "Temperature",
                "Humidity",
                "Wind Speed",
                "Solar Irradiance",
                "Panel Temperature",
                "Cloud Cover"
            ],
            "Value": [
                f"{temperature:.2f} °C",
                f"{humidity:.2f} %",
                f"{wind_speed:.2f} m/s",
                f"{solar_irradiance:.2f} W/m²",
                f"{panel_temperature:.2f} °C",
                f"{cloud_cover:.2f} %"
            ]
        })

        st.table(display_data)

    except Exception as e:

        st.error("❌ Prediction failed.")

        st.write(
            "This usually happens when the input columns used by the "
            "app do not match the columns used while training the model."
        )

        st.exception(e)


# ---------------------------------------------------------
# INFORMATION SECTION
# ---------------------------------------------------------
else:

    st.info(
        "👈 Enter the weather and solar conditions in the sidebar, "
        "then click **Predict Solar Power**."
    )

    st.subheader("📌 About This Application")

    st.write(
        """
        This application uses a machine learning model to estimate
        solar power generation based on environmental and solar
        conditions.

        **Main inputs:**
        - Temperature
        - Humidity
        - Wind Speed
        - Solar Irradiance
        - Panel Temperature
        - Cloud Cover
        """
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()

st.caption(
    "☀️ Solar Power Generation Prediction | Machine Learning Project"
)