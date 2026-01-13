# Smartphone Price Estimation System
# Dataset: Kaggle - Mobile Price Classification

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load Dataset
data = pd.read_csv("mobile_price_classification.csv")

# 2. Split Features and Target
X = data.drop("price_range", axis=1)
y = data["price_range"]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Evaluation
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print("Model Accuracy:", acc)

# 7. Save Model and Scaler
joblib.dump(model, "price_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# 8. Prediction Function (Backend Logic)
def predict_price(features):
    """
    features: list of smartphone specs in same order as dataset columns
    """
    model = joblib.load("price_model.pkl")
    scaler = joblib.load("scaler.pkl")

    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return int(prediction[0])
