from fastapi import FastAPI
import pickle
import pandas as pd


# FastAPI app
app = FastAPI(
    title="Titanic Survival Prediction API",
    description="API for Titanic survival prediction",
    version="1.0"
)


# Load trained model
with open("svc.pkl", "rb") as file:
    model = pickle.load(file)


# Home API
@app.get("/")
def home():
    return {
        "message": "Titanic Survival Prediction API is running"
    }


# Prediction API
@app.post("/predict")
def predict(
    pclass: int,
    age: float,
    sibsp: int,
    parch: int,
    fare: float,
    sex_male: int,
    embarked_Q: int,
    embarked_S: int
):

    # Input data
    input_data = pd.DataFrame(
        [[
            pclass,
            age,
            sibsp,
            parch,
            fare,
            sex_male,
            embarked_Q,
            embarked_S
        ]],
        columns=[
            "pclass",
            "age",
            "sibsp",
            "parch",
            "fare",
            "sex_male",
            "embarked_Q",
            "embarked_S"
        ]
    )

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability
    probability = model.predict_proba(input_data)[0]

    # Result
    if prediction == 1:
        result = "Passenger will survive"
    else:
        result = "Passenger will not survive"

    return {
        "prediction": int(prediction),
        "result": result,
        "survival_probability": float(probability[1]),
        "not_survival_probability": float(probability[0])
    }