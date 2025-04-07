from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# Load model and preprocessor
MODEL_PATH = "models/model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# Initialize FastAPI app
app = FastAPI(title="Insurance Enrollment Predictor")

# Input data schema
class EmployeeData(BaseModel):
    age: int
    gender: str
    marital_status: str
    salary: float
    employment_type: str
    region: str
    has_dependents: str
    tenure_years: float

@app.get("/")
def root():
    return {"message": "Welcome to the Insurance Predictor API!"}

@app.post("/predict")
def predict(data: EmployeeData):
    # Convert to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Preprocess
    processed_input = preprocessor.transform(input_df)

    # Predict
    prediction = model.predict(processed_input)[0]
    probability = model.predict_proba(processed_input)[0][1]

    return {
        "prediction": int(prediction),
        "probability": round(probability, 4)
    }
