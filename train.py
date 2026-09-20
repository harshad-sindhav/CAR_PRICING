import pandas as pd
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# 1. Titanic dataset load
df = sns.load_dataset("titanic")


# 2. Input features aur target
X = df[[
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]]

Y = df["survived"]


# 3. Missing values handle
X["age"] = X["age"].fillna(X["age"].median())
X["embarked"] = X["embarked"].fillna(X["embarked"].mode()[0])


# 4. Categorical columns ko numerical me convert
X = pd.get_dummies(
    X,
    columns=["sex", "embarked"],
    drop_first=True
)


# 5. Required 8 features
X = X[
    [
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare",
        "sex_male",
        "embarked_Q",
        "embarked_S"
    ]
]


# 6. Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42
)


# 7. Logistic Regression model
lor = LogisticRegression()

# 8. Model train
lor.fit(x_train, y_train)


# 9. Model ko pickle file me save karo
with open("svc.pkl", "wb") as file:
    pickle.dump(lor, file)


print("Model training completed!")
print("svc.pkl successfully created!")