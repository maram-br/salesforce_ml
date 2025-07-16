from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load trained model
model = joblib.load("random_forest_model.pkl")

app = FastAPI()

class OpportunityInput(BaseModel):
    owner_role: int
    opportunity_owner: int
    fiscal_period: int
    probability: float
    age: float
    days_to_close: float


@app.get("/")
def root():
    return {"message": "API is running!"}



@app.post("/predict")
def predict_opportunity(data: OpportunityInput):
    # Create DataFrame from incoming data
    input_df = pd.DataFrame([data.dict()])
    
    # Rename columns to match those used in training
    input_df.rename(columns={
        "owner_role": "Owner Role",
        "opportunity_owner": "Opportunity Owner",
        "fiscal_period": "Fiscal Period",
        "probability": "Probability (%)",
        "age": "Age",
        "days_to_close": "DaysToClose"
    }, inplace=True)
    
    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df).tolist()[0]
    
    return {
        "prediction": int(prediction),
        "probability": probability
    }
