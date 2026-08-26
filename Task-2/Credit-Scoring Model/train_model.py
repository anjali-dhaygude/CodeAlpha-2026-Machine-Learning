import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# =====================================================
# 1. CREATE DATASET
# =====================================================

data = pd.DataFrame({
    "Age": [25, 30, 35, 28, 40, 45, 23, 32, 27, 38],
    "Income": [40000, 30000, 70000, 25000, 80000, 90000, 22000, 55000, 28000, 65000],
    "LoanAmount": [10000, 15000, 20000, 12000, 25000, 30000, 8000, 18000, 10000, 22000],
    "Debt": [5000, 12000, 8000, 10000, 6000, 5000, 7000, 9000, 9000, 7000],
    "EmploymentYears": [3, 2, 8, 2, 12, 15, 1, 5, 2, 10],
    "CreditHistory": [5, 2, 9, 3, 10, 12, 1, 7, 2, 8],
    "Creditworthy": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
})


# =====================================================
# 2. SELECT FEATURES
# =====================================================

X = data[
    [
        "Age",
        "Income",
        "LoanAmount",
        "Debt",
        "EmploymentYears",
        "CreditHistory"
    ]
]


# =====================================================
# 3. SELECT TARGET
# =====================================================

y = data["Creditworthy"]


# =====================================================
# 4. SPLIT DATASET
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =====================================================
# 5. CREATE LOGISTIC REGRESSION MODEL
# =====================================================

model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(max_iter=1000))
])


# =====================================================
# 6. TRAIN MODEL
# =====================================================

model.fit(X_train, y_train)

print("Model training completed successfully!")


# =====================================================
# 7. TEST MODEL
# =====================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)


# =====================================================
# 8. SAVE MODEL
# =====================================================

joblib.dump(model, "model.pkl")

print("Model Saved Successfully!")
print("File created: model.pkl")