from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

X, y = load_iris(return_X_y=True)

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

os.makedirs("app/models", exist_ok=True)

joblib.dump(model, "app/models/iris_model.pkl")

print("Model saved.")
