import joblib
import os

MODEL_PATH = "app/models/iris_model.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found.")

    return joblib.load(MODEL_PATH)
