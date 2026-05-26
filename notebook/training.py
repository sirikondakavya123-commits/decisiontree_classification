import pandas as pd
import pickle

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# LOAD CLEANED DATASET

df = pd.read_csv(
    "../data/cleaned_loan_prediction.csv"
)

# SELECT FEATURES

X = df[
    [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Property_Area"
    ]
]

# TARGET

y = df["Loan_Status"]

# TRAIN TEST SPLIT

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# FEATURE SCALING

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)

# HYPERPARAMETER TUNING

param_grid = {
    "max_depth": [3, 5, 7, 10],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"]
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5
)

grid_search.fit(x_train, y_train)

print("Best Parameters:")

print(grid_search.best_params_)

# BEST MODEL

best_model = grid_search.best_estimator_

# PREDICTION

y_pred = best_model.predict(x_test)

# EVALUATION

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

print("\nAccuracy Score:", accuracy)

print("Precision Score:", precision)

print("Recall Score:", recall)

print("F1 Score:", f1)

# SAVE MODEL

pickle.dump(
    best_model,
    open("../models/model.pkl", "wb")
)

# SAVE SCALER

pickle.dump(
    scaler,
    open("../models/scaler.pkl", "wb")
)

print("\nModel and Scaler Saved Successfully")