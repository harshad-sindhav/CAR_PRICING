
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib

# --------------------------------
# 1. Create FastAPI application
# --------------------------------
app = FastAPI(
    title="Car Price Prediction API",
    description="API to predict used car prices",
    version="1.0.0"
)

# --------------------------------
# 2. Load trained model
# --------------------------------
MODEL_FILE = "best_model.joblib"

try:
    model = joblib.load(MODEL_FILE)
except FileNotFoundError:
    model = None


# --------------------------------
# 3. Define input data
# --------------------------------
class CarDetails(BaseModel):
    Car_Age: int = Field(ge=0, le=50)
    Kilometers: float = Field(ge=0)
    Engine_CC: float = Field(ge=500)
    Horsepower: float = Field(gt=0)
    Mileage_KMPL: float = Field(gt=0)
    Number_of_Owners: int = Field(ge=1)
    Fuel_Type: str
    Transmission: str
    Service_Count: int = Field(ge=0)
    Insurance_Years: int = Field(ge=0)
    Seats: int = Field(ge=2)
    Brand: str
    Condition: str


# --------------------------------
# 4. Category mappings
# Same as app.py and train.py
# --------------------------------
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


# --------------------------------
# 5. Home endpoint
# --------------------------------
@app.get("/")
def home():
    return {
        "message": "Car Price Prediction API is running"
    }


# --------------------------------
# 6. Prediction endpoint
# --------------------------------
@app.post("/predict")
def predict_price(car: CarDetails):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model file best_model.joblib nahi mili. Pehle train.py run karo."
        )

    if car.Fuel_Type not in fuel_mapping:
        raise HTTPException(
            status_code=400,
            detail="Invalid Fuel_Type"
        )

    if car.Transmission not in transmission_mapping:
        raise HTTPException(
            status_code=400,
            detail="Invalid Transmission"
        )

    if car.Brand not in brand_mapping:
        raise HTTPException(
            status_code=400,
            detail="Invalid Brand"
        )

    if car.Condition not in condition_mapping:
        raise HTTPException(
            status_code=400,
            detail="Invalid Condition"
        )

    input_data = pd.DataFrame([{
        "Car_Age": car.Car_Age,
        "Kilometers": car.Kilometers,
        "Engine_CC": car.Engine_CC,
        "Horsepower": car.Horsepower,
        "Mileage_KMPL": car.Mileage_KMPL,
        "Number_of_Owners": car.Number_of_Owners,
        "Fuel_Type": fuel_mapping[car.Fuel_Type],
        "Transmission": transmission_mapping[car.Transmission],
        "Service_Count": car.Service_Count,
        "Insurance_Years": car.Insurance_Years,
        "Seats": car.Seats,
        "Brand": brand_mapping[car.Brand],
        "Condition": condition_mapping[car.Condition]
    }])

    try:
        prediction = model.predict(input_data)[0]

        return {
            "predicted_price_lakh": round(float(prediction), 2),
            "currency": "INR",
            "unit": "Lakh"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )
