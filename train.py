
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# --------------------------------
# 1. Load dataset
# --------------------------------
DATASET_FILE = "car_price_500_rows_14_columns.csv"

df = pd.read_csv(DATASET_FILE)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print(df.head())

# --------------------------------
# 2. Encode categorical columns
# Same mappings as app.py
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
# 3. Define input features
# --------------------------------
feature_columns = [
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
]

target_column = "Car_Price_Lakh"

# Check required columns
required_columns = feature_columns + [target_column]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Dataset mein ye columns missing hain: {missing_columns}"
    )

# --------------------------------
# 4. Apply categorical mappings
# --------------------------------
df["Fuel_Type"] = df["Fuel_Type"].map(fuel_mapping)

df["Transmission"] = df["Transmission"].map(
    transmission_mapping
)

df["Brand"] = df["Brand"].map(brand_mapping)

df["Condition"] = df["Condition"].map(
    condition_mapping
)

# Convert columns to numeric and handle missing values
for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required_columns).copy()

if df.empty:
    raise ValueError(
        "Encoding ke baad dataset empty hai. "
        "Dataset ke column names aur category values check karo."
    )

# --------------------------------
# 5. Split X and y
# --------------------------------
X = df[feature_columns]
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# --------------------------------
# 6. Train model
# --------------------------------
model = LinearRegression()

model.fit(X_train, y_train)

print("Model training completed!")

# --------------------------------
# 7. Evaluate model
# --------------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", round(mae, 3))
print("R2 Score:", round(r2, 4))

# --------------------------------
# 8. Save trained model
# --------------------------------
joblib.dump(model, "best_model.joblib")

print("Model saved as best_model.joblib")
print("Ab Streamlit app.py run kar sakte ho.")
