# dashboard.py
# ==============================================================================
# This script creates a user-friendly web dashboard for our Bearing RUL Predictor.
# It provides a file uploader and communicates with our deployed FastAPI backend.
# ==============================================================================

import streamlit as st
import requests
import pandas as pd
from typing import List

# --- Configuration ---
#
# CRITICAL: Replace the placeholder URL below with the actual URL of YOUR
# live application deployed on Render. Make sure to include the "/predict" at the end.
#
API_URL = "https://bearing-remaining-lifetime-prediction.onrender.com/predict"  # <-- Example URL. YOU MUST CHANGE THIS!

# We know from our model training that it requires a history of 15 readings.
WINDOW_SIZE = 15

# --- Webpage Interface Setup ---
st.set_page_config(page_title="Bearing RUL Predictor", layout="wide")

st.title("⚙️ Bearing Remaining Useful Life (RUL) Predictor")
st.write(
    "This dashboard provides a user-friendly interface to interact with our deployed AI model. "
    "Upload a sequence of bearing vibration data files to get a prediction."
)
st.markdown("---")

# --- File Uploader and Logic ---
st.header("1. Upload Your Data")

# Create a file uploader that accepts multiple files.
uploaded_files = st.file_uploader(
    f"Upload exactly {WINDOW_SIZE} sequential bearing data files.",
    accept_multiple_files=True
)

# The main logic only runs if files have been uploaded.
if uploaded_files:
    # --- Validation Step ---
    # First, check if the user has uploaded the correct number of files.
    if len(uploaded_files) != WINDOW_SIZE:
        st.warning(f"⚠️ Please upload exactly {WINDOW_SIZE} files. You have uploaded {len(uploaded_files)}.")
    else:
        # If the number is correct, show the names of the uploaded files.
        st.success(f"Successfully uploaded {len(uploaded_files)} files. Ready to predict.")
        
        # --- Prediction Trigger ---
        st.header("2. Get Prediction")
        
        # Create a button to start the prediction process.
        if st.button("Predict RUL", type="primary"):
            
            # Use a spinner to show the user that something is happening.
            with st.spinner("Processing files and connecting to AI model..."):
                
                # --- Data Processing ---
                # Convert the uploaded files into the format our API expects.
                signal_sequence = []
                error_parsing = False
                for file in uploaded_files:
                    try:
                        # Use pandas to easily read the tab-separated data.
                        df = pd.read_csv(file, sep='\t', header=None)
                        # Extract the first column (our signal) and convert to a list.
                        signal = df[0].tolist()
                        signal_sequence.append(signal)
                    except Exception as e:
                        st.error(f"Error reading file '{file.name}': {e}")
                        error_parsing = True
                        break

                # --- API Call ---
                if not error_parsing:
                    # Prepare the JSON payload for the API request.
                    payload = {"signals": signal_sequence}
                    
                    try:
                        # Make the POST request to our deployed API on Render.
                        response = requests.post(API_URL, json=payload, timeout=30)

                        if response.status_code == 200:
                            # If successful, extract the RUL and display it.
                            result = response.json()
                            predicted_rul_cycles = result.get("predicted_rul")
                            predicted_rul_hours = (predicted_rul_cycles * 10) / 60


                            st.subheader("Prediction Result")
                            
                            # Safely handle the case where the key might be missing
                            if predicted_rul_hours is not None:
                                st.metric(
                                    label="Predicted Remaining Useful Life (Hours)", 
                                    value=f"{predicted_rul_hours:.1f}"
                                )
                            else:
                                st.error("API returned a successful response, but the 'predicted_rul' key was missing.")
                                st.json(result)
                        else:
                            # If the API returns an error, display it for debugging.
                            st.error(f"API Error: Received status code {response.status_code}")
                            st.json(response.json())

                    except requests.exceptions.RequestException as e:
                        # If we can't connect to the API at all.
                        st.error(f"Connection Error: Could not connect to the API. Is it deployed and running? Details: {e}")