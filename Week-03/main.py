
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title='Steel Industry Energy Prediction')

# Setup static files
os.makedirs('static', exist_ok=True)
app.mount('/static', StaticFiles(directory='static'), name='static')

# Load the Pipeline
model = joblib.load('week3_energy_model.joblib')

# Define feature names to avoid the UserWarning
FEATURE_NAMES = [
    'Lagging_Current_Reactive.Power_kVarh', 
    'Leading_Current_Reactive_Power_kVarh', 
    'CO2(tCO2)', 
    'Lagging_Current_Power_Factor', 
    'Leading_Current_Power_Factor', 
    'NSM', 
    'WeekStatus', 
    'Day_of_week', 
    'Load_Type'
]

class EnergyInput(BaseModel):
    Lagging_Current_Reactive_Power_kVarh: float
    Leading_Current_Reactive_Power_kVarh: float
    CO2_tCO2: float
    Lagging_Current_Power_Factor: float
    Leading_Current_Power_Factor: float
    NSM: int
    WeekStatus: int
    Day_of_week: int
    Load_Type: int

@app.get('/', response_class=HTMLResponse)
def read_root():
    with open('templates/index.html', 'r') as f:
        return f.read()

@app.post('/predict')
def predict(data: EnergyInput):
    # We create a DataFrame with feature names to satisfy Scikit-Learn and remove warnings
    input_df = pd.DataFrame([[data.Lagging_Current_Reactive_Power_kVarh,
                              data.Leading_Current_Reactive_Power_kVarh,
                              data.CO2_tCO2,
                              data.Lagging_Current_Power_Factor,
                              data.Leading_Current_Power_Factor,
                              data.NSM,
                              data.WeekStatus,
                              data.Day_of_week,
                              data.Load_Type]], columns=FEATURE_NAMES)
    
    prediction = model.predict(input_df)
    return {'predicted_usage_kWh': float(prediction[0])}
