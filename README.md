
# 🛠️ Bearing Lifetime Prediction – End-to-End ML Application

🔗 [Live Dashboard](https://bearing-remaining-lifetime-prediction-rtj.streamlit.app/) | 🌐 [Backend API](https://bearing-remaining-lifetime-prediction.onrender.com) | 👨‍💼 [LinkedIn](https://www.linkedin.com/in/vaibhav-admane-15rtj)

A containerized machine learning pipeline that predicts the **Remaining Useful Life (RUL)** of industrial bearings using raw vibration data. The system includes preprocessing, wavelet-based feature engineering, XGBoost modeling, and deployment as a FastAPI service + Streamlit dashboard.

---

## 📌 Table of Contents

- [📍 Problem Statement](#-problem-statement)
- [✅ Solution Overview](#-solution-overview)
- [🏗️ Project Architecture](#-project-architecture)
- [📁 Project Structure](#-project-structure)
- [🔬 Methodology](#-methodology)
- [🧰 Technology Stack](#-technology-stack)
- [⚙️ Local Setup & Execution](#-local-setup--execution)
  - [▶️ Backend API (Docker)](#1-backend-api-docker)
  - [🎛️ Frontend Dashboard (Streamlit)](#2-frontend-dashboard-streamlit)
- [🔗 API Endpoint Details](#-api-endpoint-details)
- [🚀 Deployment Strategy](#-deployment-strategy)
- [📊 How to Use the Live Dashboard](#-how-to-use-the-live-dashboard)
- [📎 References](#-references)

---

## 📍 Problem Statement

Unexpected failures of rotating bearings in industrial systems can lead to major downtime, safety risks, and maintenance costs. Traditional approaches rely on reactive or scheduled maintenance, which aren't optimal.

This project aims to shift from reactive to **predictive maintenance** by forecasting the remaining life of a bearing using sensor data and machine learning.

---

## ✅ Solution Overview

This solution consists of two modular services:

1. **⚙️ FastAPI Backend**: Accepts raw vibration data, extracts features, and predicts RUL using a trained model.
2. **🖥️ Streamlit Dashboard**: A user-friendly UI where users can upload data files and visualize RUL predictions.

---

## 🏗️ Project Architecture

```text
User → Streamlit Dashboard → FastAPI Backend → Model Inference → RUL Prediction → Display
```

**Deployment Overview:**
- ✅ Backend: Deployed to Render as a Docker container.
- ✅ Frontend: Hosted on Streamlit Community Cloud.

---

## 📁 Project Structure

```text
/bearing-lifetime-prediction/
├── data/                       # IMS raw bearing data (external)
├── notebooks/
│   └── model_dev.ipynb         # Wavelet + training exploration
├── bearing_app/
│   ├── assets/
│   │   ├── model.json
│   │   └── config.json
│   ├── __init__.py
│   ├── main.py                 # FastAPI entry point
│   └── prognosticator.py       # RUL prediction logic
├── dashboard.py                # Streamlit app
├── Dockerfile                  # Container blueprint
├── requirements.txt            # Backend dependencies
├── requirements-dashboard.txt  # Frontend dependencies
└── README.md                   # Project documentation
```

---

## 🔬 Methodology

1. **Wavelet Filtering**  
   - Based on *Qiu et al. (2006)*, we use **Morlet Wavelet Transform** to extract fault signatures from noisy vibration signals.
   - Parameters `α` and `β` optimized using Shannon Entropy + SVD periodicity scoring.

2. **Model Training**  
   - Sliding window (size = 15) used to train an **XGBoost Regressor** on Health Indicator (HI) curves.

3. **Validation**  
   - Model trained on `2nd_test` and `3rd_test`, and validated on unseen `1st_test` dataset.

4. **Deployment**  
   - Backend containerized using Docker and deployed on Render.
   - Streamlit dashboard interacts with backend via HTTP requests.

---

## 🧰 Technology Stack

| Layer         | Technologies                             |
|---------------|-------------------------------------------|
| **Backend**   | Python, FastAPI, Uvicorn                  |
| **Frontend**  | Streamlit                                 |
| **ML & DSP**  | XGBoost, Scikit-learn, PyWavelets, SciPy  |
| **Data**      | NumPy, Pandas                             |
| **Container** | Docker                                    |
| **Cloud**     | Render (Backend), Streamlit Cloud (UI)    |

---

## ⚙️ Local Setup & Execution

### ▶️ 1. Backend API (Docker)

> **Requirements:** Docker Desktop installed

```bash
# Clone repo
git clone https://github.com/YourUsername/bearing-lifetime-prediction.git
cd bearing-lifetime-prediction

# Build image
docker build -t bearing-predictor-app .

# Run container
docker run -p 8000:8000 --name bearing-api -d bearing-predictor-app

# API will be available at:
# http://localhost:8000/docs
```

---

### 🎛️ 2. Frontend Dashboard (Streamlit)

> **Requirements:** Python 3.9+ and virtualenv

```bash
# From project root
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scriptsctivate

# Install dependencies
pip install -r requirements-dashboard.txt

# Run Streamlit app
streamlit run dashboard.py
```

> **Make sure:** `API_URL` in `dashboard.py` points to the correct FastAPI backend (local or deployed)

---

## 🔗 API Endpoint Details

### `POST /predict`

- **URL:** `https://bearing-remaining-lifetime-prediction.onrender.com/predict`
- **Method:** `POST`
- **Payload:**

```json
{
  "signals": [
    [0.1, -0.2, 0.1, ...],
    [0.2, -0.1, 0.2, ...],
    ...
    [0.0, 0.3, -0.1, ...]
  ]
}
```

> Must contain exactly **15 signal windows** (sliding input)

- **Response:**

```json
{
  "predicted_rul": 145.82,
  "status": "success"
}
```

> Output is in **remaining cycles/files** — converted to hours in the dashboard.

---

## 🚀 Deployment Strategy

| Service   | Platform           | Description                                |
|-----------|--------------------|--------------------------------------------|
| Backend   | Render             | Dockerized FastAPI container as Web Service |
| Frontend  | Streamlit Cloud    | Public Streamlit dashboard auto-updated    |

---

## 📊 How to Use the Live Dashboard

1. Visit the deployed [Streamlit Dashboard](https://bearing-remaining-lifetime-prediction-rtj.streamlit.app/).
2. Click **"Browse files"** to upload **15 sequential vibration files**.
3. Click **"Predict RUL"**.
4. View the remaining life in **hours**.

---

## 📎 References

- **Qiu, H., Lee, J., Lin, J., & Yu, G. (2006)**  
  *Wavelet filter-based weak signature detection method and its application on rolling element bearing prognostics.*  
  *Journal of Sound and Vibration, 289(4–5), 1066–1090.*

- **IMS Bearing Dataset**  
  [NASA Prognostics Data Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)
