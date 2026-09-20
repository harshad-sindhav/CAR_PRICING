import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# -----------------------------
# Load trained model
# -----------------------------
MODEL_FILE = "best_model.joblib"

try:
    model = joblib.load(MODEL_FILE)
except FileNotFoundError:
    st.error("best_model.joblib file nahi mili. Is file ko app.py ke same folder me rakho.")
    st.stop()

# -----------------------------
# LabelEncoder mappings used in the notebook
# The notebook uses LabelEncoder separately on each column.
# -----------------------------
fuel_mapping = {
    "CNG": 0,
    "Diesel": 1,
    "Electric": 2,
    "Petrol": 3
}

transmission_mapping = {
    "Automatic": 0,
    "Manual": 1
}

brand_mapping = {
    "Ford": 0,
    "Honda": 1,
    "Hyundai": 2,
    "Kia": 3,
    "Mahindra": 4,
    "Maruti": 5,
    "Tata": 6,
    "Toyota": 7
}

condition_mapping = {
    "Average": 0,
    "Excellent": 1,
    "Good": 2,
    "Poor": 3
}

# -----------------------------
# UI
# -----------------------------
st.title("🚗 Car Price Prediction")
st.write("Enter the car details below to predict its price.")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    car_age = st.number_input(
        "Car Age (years)",
        min_value=0,
        max_value=50,
        value=5,
        step=1
    )

    kilometers = st.number_input(
        "Kilometers",
        min_value=0,
        max_value=1000000,
        value=50000,
        step=1000
    )

    engine_cc = st.number_input(
        "Engine CC",
        min_value=500,
        max_value=10000,
        value=1500,
        step=100
    )

    horsepower = st.number_input(
        "Horsepower",
        min_value=20,
        max_value=1000,
        value=100,
        step=5
    )

    mileage = st.number_input(
        "Mileage (KMPL)",
        min_value=1.0,
        max_value=100.0,
        value=18.0,
        step=0.1
    )

with col2:
    owners = st.number_input(
        "Number of Owners",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

    fuel_type = st.selectbox(
        "Fuel Type",
        list(fuel_mapping.keys())
    )

    transmission = st.selectbox(
        "Transmission",
        list(transmission_mapping.keys())
    )

    service_count = st.number_input(
        "Service Count",
        min_value=0,
        max_value=50,
        value=5,
        step=1
    )

    insurance_years = st.number_input(
        "Insurance Years",
        min_value=0,
        max_value=20,
        value=2,
        step=1
    )

with col3:
    seats = st.number_input(
        "Seats",
        min_value=2,
        max_value=15,
        value=5,
        step=1
    )

    brand = st.selectbox(
        "Brand",
        list(brand_mapping.keys())
    )

    condition = st.selectbox(
        "Condition",
        list(condition_mapping.keys())
    )

st.divider()

if st.button("🔮 Predict Car Price", type="primary", use_container_width=True):

    # Same 13 feature order used by the notebook:
    # Car_Age, Kilometers, Engine_CC, Horsepower, Mileage_KMPL,
    # Number_of_Owners, Fuel_Type, Transmission, Service_Count,
    # Insurance_Years, Seats, Brand, Condition

    input_data = pd.DataFrame([[
        car_age,
        kilometers,
        engine_cc,
        horsepower,
        mileage,
        owners,
        fuel_mapping[fuel_type],
        transmission_mapping[transmission],
        service_count,
        insurance_years,
        seats,
        brand_mapping[brand],
        condition_mapping[condition]
    ]], columns=[
        "Car_Age",
        "Kilometers",
        "Engine_CC",
        "Horsepower",
        "Mileage_KMPL",
        "Number_of_Owners",
        "Fuel_Type",
        "Transmission",
        "Service_Count",
        "Insurance_Years",
        "Seats",
        "Brand",
        "Condition"
    ])

    try:
        prediction = model.predict(input_data)[0]

        st.success("Prediction completed successfully!")
        st.metric(
            label="Predicted Car Price",
            value=f"₹ {prediction:.2f} Lakh"
        )

        st.info(f"Approximately ₹ {prediction * 100:.2f} Crore equivalent is NOT used here; "
                f"the model output is directly in Lakh as in the notebook.")

    except Exception as e:
        st.error(f"Prediction me error aaya: {e}")

st.caption("Model: Linear Regression | Target: Car_Price_Lakh")

