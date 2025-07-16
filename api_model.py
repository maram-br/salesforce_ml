from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the full pipeline that includes preprocessing and model
pipeline = joblib.load("random_forest_pipeline.pkl")  # your saved pipeline file

app = FastAPI()

class OpportunityInput(BaseModel):
    Account_Name: str
    Opportunity_Owner: str
    DRX_Carline: str
    Carline_Segment: str
    E_Mobility_Category: str
    Project_Category: str

@app.get("/")
def root():
    return {"message": "API is running!"}

@app.post("/predict")
def predict_opportunity(data: OpportunityInput):
    # Create DataFrame with columns named exactly as in training
    input_df = pd.DataFrame([{
        "Account Name: Account Name": data.Account_Name,
        "Opportunity Owner: Full Name": data.Opportunity_Owner,
        "DRX Carline: DRX Carline Name": data.DRX_Carline,
        "Carline Segment": data.Carline_Segment,
        "E-Mobility Category": data.E_Mobility_Category,
        "Project Category": data.Project_Category
    }])
    
    # Predict using pipeline (which does encoding internally)
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]  # probability for class 1 (IsWon)
    
    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }
