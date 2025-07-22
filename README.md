# Bearing Lifetime Prediction API

This project is an end-to-end machine learning application designed to predict the Remaining Useful Life (RUL) of rolling element bearings. It uses a sophisticated wavelet-based feature engineering approach from raw vibration data, trains a predictive model, and deploys it as a containerized web API.

## Table of Contents
- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Technology Stack](#technology-stack)
- [How to Run Locally](#how-to-run-locally)
- [API Endpoint](#api-endpoint)
- [How to Deploy](#how-to-deploy)

---

## Problem Statement

In manufacturing and industrial settings, the unexpected failure of bearings in critical machinery can lead to costly downtime, production loss, and safety hazards. The goal of this project is to move from reactive maintenance (fixing things after they break) to **prognostic maintenance** (predicting when a failure will occur). By accurately predicting the RUL of a bearing, maintenance can be scheduled proactively, maximizing uptime and efficiency.

---

## Solution Overview

This application provides a REST API that accepts a sequence of raw vibration signals from a bearing and returns a prediction of its Remaining Useful Life in operational cycles.

The core of the solution is a two-part process:
1.  **Feature Engineering:** A custom-tuned Morlet wavelet filter is used to extract a "Health Indicator" (HI) from the noisy, raw vibration data. This technique is based on the paper "Wavelet Filter-based Weak Signature Detection Method and its Application on Rolling Element Bearing Prognostics" by Qiu et al.
2.  **Predictive Modeling:** An XGBoost regression model is trained on the HI data from multiple run-to-failure experiments to learn the pattern of degradation over time.

---

## Project Structure

The project is organized into distinct directories for clarity and maintainability.

/
|--- data/ # Raw IMS Bearing Data
|--- notebooks/ # Jupyter notebooks for experimentation (Phase 1)
|--- bearing_app/ # The deployable application source code
| |--- assets/ # Contains the trained model and config files
| |--- init.py
| |--- main.py # FastAPI application and API endpoints
| |--- prognosticator.py # Core class for feature extraction and prediction
| --- requirements.txt # Application dependencies 
|--- .gitignore # Files and folders for Git to ignore 
|--- Dockerfile # Blueprint for building the application container
|--- README.md # This file


---

## Methodology

1.  **Data Source:** The project uses the [NASA IMS Bearing Dataset](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository), which contains run-to-failure vibration data for several bearings.
2.  **Feature Extraction:** An optimal Morlet wavelet filter is designed by finding the `β` (shape) and `α` (scale) parameters that best isolate periodic, impulse-like fault signatures from a sample faulty signal. The Root Mean Square (RMS) of the filtered signal serves as the Health Indicator.
3.  **Model Training:** The model is trained on the Health Indicator curves from two full experiments (`2nd_test` and `3rd_test`). A sliding window approach is used to provide the model with historical context for each prediction.
4.  **Validation:** The trained model is validated on a completely unseen experiment (`1st_test`) to ensure it generalizes well to new data.
5.  **Deployment:** The final application is containerized using Docker and is ready for deployment on any cloud platform that supports containers (e.g., Render, IBM Cloud Code Engine, Google Cloud Run).

---

## Technology Stack

*   **Backend:** Python 3.11
*   **API Framework:** FastAPI
*   **Machine Learning:** Scikit-learn, XGBoost
*   **Data Processing:** NumPy, Pandas, SciPy
*   **Signal Processing:** PyWavelets
*   **Containerization:** Docker

---

## How to Run Locally

To run the application on your local machine for development or testing.

**Prerequisites:**
*   Python 3.11+
*   Docker Desktop (running)

**1. Clone the Repository**
```bash
git clone https://github.com/YourUsername/bearing-lifetime-prediction.git
cd bearing-lifetime-prediction

