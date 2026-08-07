from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Heart Disease Prediction API")

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "best_model.joblib")
SCALER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "scaler.joblib")

# Load model and scaler at startup
model = None
scaler = None
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
except Exception as e:
    print(f"Error loading model or scaler: {e}")

class PredictionRequest(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

@app.get("/")
def home():
    return {"message": "Welcome to the Heart Disease Prediction API", "model_loaded": model is not None, "scaler_loaded": scaler is not None}

@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler is not loaded")
    
    # Create DataFrame from input
    input_df = pd.DataFrame([request.model_dump()])
    
    # Scale input data
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
    
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[:, 1] if hasattr(model, "predict_proba") else None
    
    return {
        "prediction": int(prediction[0]),
        "probability": float(probability[0]) if probability is not None else None
    }
