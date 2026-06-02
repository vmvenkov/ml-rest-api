from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.model import load_model

app = FastAPI(title="ML REST API", version="1.0")

try:
    model = load_model()
except Exception:
    model = None


class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)


@app.get("/")
def home():
    return {"message": "ML REST API is running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(data: IrisInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    try:
        features = [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]]

        prediction = model.predict(features)[0]

        labels = {
            0: "setosa",
            1: "versicolor",
            2: "virginica"
        }

        return {
            "prediction": int(prediction),
            "class_name": labels[int(prediction)]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )
