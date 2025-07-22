# main.py
# ==============================================================================
# This file defines the API endpoints for our RUL prediction service using FastAPI.
# This is the "Front Door" to our application. It handles web requests and
# uses the 'BearingPrognosticator' class to perform the actual work.
# ==============================================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import json
import os

# Import our prediction engine
from prognosticator import BearingPrognosticator

# --- 1. Initialize the FastAPI App ---
app = FastAPI(
    title="Bearing RUL Prediction API",
    description="An API to predict the Remaining Useful Life (RUL) of a bearing using a trained ML model.",
    version="1.0.0"
)

# --- 2. Load Model and Config at Startup ---
# This is efficient because the model and config are loaded only once when the
# application starts, not for every single request.
MODEL_PATH = "Assets/model.json"
CONFIG_PATH = "Assets/config.json"

if not os.path.exists(MODEL_PATH) or not os.path.exists(CONFIG_PATH):
    raise RuntimeError("FATAL: model.json or config.json not found. Please run Phase 1 first.")

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

# Instantiate our prediction engine
predictor = BearingPrognosticator(model_path=MODEL_PATH, config=config)


# --- 3. Define the Request Data Model ---
# Pydantic models provide data validation and documentation automatically.
# This defines what the JSON input for our API should look like.
class PredictionRequest(BaseModel):
    # A list of lists, where each inner list is a raw signal from one time step.
    signals: List[List[float]] 
    
    class Config:
        schema_extra = {
            "example": {
                "signals": [
                    [0.1, -0.2, 0.3, "..."], # Signal at time t-14
                    [0.5, 0.1, -0.1, "..."], # Signal at time t-13
                    # ... 13 more signals ...
                ]
            }
        }


# --- 4. Define the API Endpoint ---
@app.post("/predict")
def predict_rul(request: PredictionRequest):
    """
    Receives a sequence of raw vibration signals and returns the predicted RUL.
    
    The input must be a JSON object with a key "signals", which is a list
    containing a number of inner lists equal to the model's `window_size`.
    """
    try:
        # The main logic: call the predict_rul method of our engine class
        predicted_rul = predictor.predict_rul(request.signals)
        
        # Return the prediction in a JSON response
        return {"predicted_rul": predicted_rul, "status": "success"}

    except ValueError as e:
        # Handle known errors, like incorrect window size
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle any other unexpected errors during prediction
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during prediction.")

@app.get("/")
def read_root():
    """A simple root endpoint to confirm the API is running."""
    return {"message": "Welcome to the Bearing RUL Prediction API. Please POST to /predict to get a prediction."}