# 🚗 Car Price Prediction

A Machine Learning project that predicts the estimated price of a car based on its specifications and condition. The project uses Python, Scikit-learn, Streamlit, and FastAPI to demonstrate machine learning model development and deployment.

## 📌 Project Overview

The Car Price Prediction project takes car-related information as input and predicts an estimated selling price using a trained Machine Learning regression model.

The project includes a web interface for interactive predictions and an API component for serving model predictions.

## ✨ Features

- Predict estimated car prices using a trained regression model.
- Interactive web interface built with Streamlit.
- REST API built with FastAPI.
- Input features include car age, kilometers driven, engine capacity, horsepower, mileage, fuel type, transmission, brand, and condition.
- Saved model loading for predictions.
- Model evaluation using regression metrics.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- FastAPI
- Pydantic
- Joblib
- Uvicorn
- Git and GitHub

## 📂 Project Structure

```text
CAR_PRICING/
├── app.py
├── api.py
├── train.py
├── best_model.joblib
├── requirements.txt
├── README.md
└── car_data.csv
```

*Note: The project structure above is illustrative. Include `car_data.csv` only if the dataset is available in your project folder or repository.*

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/harshad-sindhav/CAR_PRICING.git
```

### 2. Navigate to the project directory

```bash
cd CAR_PRICING
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## ▶️ Run the Application

### Streamlit Web App

```bash
streamlit run app.py
```

### FastAPI Server

```bash
uvicorn api:app --reload
```

Once the API server starts, open the following URL in your browser to explore the API documentation:

```text
http://127.0.0.1:8000/docs
```

## 🤖 Machine Learning Workflow

1. Load and prepare the car price dataset.
2. Handle missing values and preprocess input features.
3. Split the dataset into training and testing sets.
4. Train a regression model.
5. Evaluate model performance using suitable regression metrics.
6. Save the trained model.
7. Use the saved model to generate predictions through the application and API.

## 📊 Model Evaluation

The model can be evaluated using:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted prices.
- **RMSE (Root Mean Squared Error):** Measures prediction error while penalizing larger errors.
- **R² Score:** Measures how well the model explains variation in the target values.

Actual evaluation scores should be added here after running the model evaluation.

## 🎯 Learning Outcomes

- Understanding supervised machine learning and regression.
- Building a prediction interface with Streamlit.
- Creating REST APIs with FastAPI.
- Saving and loading trained machine learning models.
- Managing dependencies and project files.
- Using GitHub for version control and project presentation.

## 👨‍💻 Author

**Harshad Sindhav**

GitHub: [harshad-sindhav](https://github.com/harshad-sindhav)

## 📌 Future Improvements

- Improve prediction performance through model comparison and hyperparameter tuning.
- Add more vehicle features and data validation.
- Improve the user interface and API error handling.
- Deploy the application for public access.

---

*This project was developed as part of my learning journey in Python, Machine Learning, and application development.*
