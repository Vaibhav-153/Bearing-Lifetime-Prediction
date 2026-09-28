# Bearing Lifetime Prediction

This project estimates the Remaining Useful Life (RUL) of rolling bearings from vibration data in the IMS bearing dataset. It uses a wavelet-based health indicator, an XGBoost regression model, a FastAPI prediction service, and a Streamlit dashboard.

The repository is mainly a learning and portfolio project. The model is not intended for production maintenance decisions without wider validation on more bearings and operating conditions.

## Problem statement

Bearing failures can cause unplanned machine downtime. Vibration signals change as a bearing degrades, so a run-to-failure signal history can be used to estimate how many acquisition cycles remain before the end of the recorded run.

The goal of this project is to:

1. extract a simple degradation indicator from raw vibration snapshots,
2. use recent indicator values as a time window,
3. train a regression model to estimate remaining cycles,
4. expose the trained model through an API and a small web interface.

## Dataset

The project uses the **IMS Bearings** dataset from the Center for Intelligent Maintenance Systems, University of Cincinnati, distributed through the NASA Prognostics Data Repository.

Dataset page:
https://data.nasa.gov/dataset/ims-bearings

The training script expects the extracted data in this layout:

```text
Data/
├── 1st_test/
├── 2nd_test/
└── 3rd_test/
```

The original IMS files contain chronological vibration snapshots from run-to-failure bearing experiments. This project works with selected signal columns from those runs.

## Method

### 1. Health indicator

For each vibration snapshot:

- apply a Morlet continuous wavelet transform (`morl`),
- keep the coefficient values at one selected scale,
- calculate the RMS of the real coefficients.

That RMS value is used as the Health Indicator (HI) for the snapshot.

The previous version of this project described both `alpha` and `beta` as optimized wavelet parameters. In the actual PyWavelets implementation, only the scale was searched. The unused beta parameter has therefore been removed from the code and configuration.

### 2. Wavelet scale selection

The wavelet scale is selected using a late-stage signal from the `2nd_test` training run. Candidate scales from 1 to 10 are evaluated with a simple SVD-based periodicity score, and the scale with the highest score is selected.

The saved model currently uses a scale of approximately:

```text
6.6842105263
```

### 3. Sliding windows

The model uses 15 consecutive HI values as one input sample.

For a run containing `N` snapshots, the target RUL is a countdown in dataset cycles:

```text
N-1, N-2, ..., 2, 1, 0
```

Each training row therefore contains:

```text
[HI(t-14), HI(t-13), ..., HI(t)] -> remaining cycles
```

### 4. Training and validation split

The current experiment keeps the original project split:

| Purpose | IMS run | Signal column |
| --- | --- | ---: |
| Training | `2nd_test` | 0 |
| Training | `3rd_test` | 2 |
| Validation | `1st_test` | 7 |

The validation run is not included in model fitting.

This is a single held-out run, not full cross-validation across all bearings. The reported metrics should be interpreted with that limitation in mind.

## Model

The regression model is `XGBRegressor` with the following main settings:

```text
n_estimators=1000
learning_rate=0.05
max_depth=5
subsample=0.8
colsample_bytree=0.8
random_state=42
```

The trained XGBoost model is stored as JSON under `bearing_app/Assets/model.json`.

## Results

The original notebook recorded these results on the selected `1st_test` validation signal:

| Metric | Result |
| --- | ---: |
| RMSE | 1022.24 cycles |
| MAE | 856.13 cycles |

These numbers are useful as a reproducible baseline, but they also show that the current model still has substantial prediction error. The project should not be presented as an accurate industrial RUL system yet.

Running the training script writes the current metrics to:

```text
reports/metrics.json
```

## Project structure

```text
Bearing-Lifetime-Prediction/
├── .github/
│   └── workflows/
│       └── ci.yml
├── Data/
│   ├── 1st_test/
│   ├── 2nd_test/
│   └── 3rd_test/
├── bearing_app/
│   ├── Assets/
│   │   ├── config.json
│   │   └── model.json
│   ├── __init__.py
│   ├── features.py
│   ├── main.py
│   ├── prognosticator.py
│   └── requirements.txt
├── notebooks/
│   └── model_development.ipynb
├── reports/
├── scripts/
│   ├── __init__.py
│   └── train_model.py
├── tests/
│   ├── test_api.py
│   ├── test_features.py
│   ├── test_prognosticator.py
│   └── test_training.py
├── .dockerignore
├── .gitignore
├── dashboard.py
├── Dockerfile
├── LICENSE
├── pyproject.toml
├── requirements-dev.txt
├── requirements-training.txt
├── requirements.txt
├── render.yaml
└── README.md
```

## Installation

Python 3.11 is used by the Docker image and CI workflow.

### Dashboard only

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Training environment

```bash
python -m pip install -r requirements-training.txt
```

### Development and tests

```bash
python -m pip install -r requirements-dev.txt
```

## Train the model

Make sure the IMS folders exist under `Data/`, then run:

```bash
python -m scripts.train_model
```

The script will:

1. load the selected training signal columns,
2. select the wavelet scale using training data,
3. calculate health indicators,
4. create 15-step sliding windows,
5. train the XGBoost model,
6. evaluate on the held-out validation run,
7. save the model and configuration,
8. write validation metrics to `reports/metrics.json`.

## Run the API locally

Install the API dependencies:

```bash
python -m pip install -r bearing_app/requirements.txt
```

Start FastAPI:

```bash
python -m uvicorn bearing_app.main:app --reload --port 8000
```

Useful URLs:

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

## Prediction API

Endpoint:

```text
POST /predict
```

Request format:

```json
{
  "signals": [
    [0.10, -0.20, 0.30],
    [0.11, -0.18, 0.28]
  ]
}
```

The production model expects exactly 15 signal snapshots. Each inner list represents one raw vibration snapshot.

Successful response:

```json
{
  "predicted_rul": 145.82,
  "status": "success"
}
```

The prediction is returned in dataset cycles.

## Run the Streamlit dashboard

```bash
streamlit run dashboard.py
```

By default, the dashboard uses the deployed Render API URL. To use a local backend, set `BEARING_API_URL` before starting Streamlit.

Windows PowerShell:

```powershell
$env:BEARING_API_URL="http://127.0.0.1:8000/predict"
streamlit run dashboard.py
```

Linux/macOS:

```bash
export BEARING_API_URL="http://127.0.0.1:8000/predict"
streamlit run dashboard.py
```

The dashboard sorts uploaded files by filename before sending them to the API so the sequence stays chronological.

## Docker

Build the API image:

```bash
docker build -t bearing-rul-api .
```

Run it locally:

```bash
docker run --rm -p 8000:8000 bearing-rul-api
```

The container uses port `8000` locally and respects the `PORT` environment variable on hosting platforms. A small `render.yaml` is included so the API health check and Docker deployment settings are versioned with the project.

## Tests and code quality

Run the tests:

```bash
python -m pytest -q
```

Run Ruff:

```bash
ruff check .
```

GitHub Actions runs both checks on pushes and pull requests.

## Limitations

- The model is trained on only two selected signal columns and validated on one held-out signal column.
- The target is a countdown based on file position, not an independently measured physical RUL value.
- The wavelet scale is selected from one training run.
- XGBoost hyperparameters are fixed rather than selected through run-level cross-validation.
- The dashboard assumes the first column of each uploaded file is the signal to predict from.
- The hours shown in the dashboard assume one dataset cycle represents 10 minutes.
- The model does not provide prediction intervals or uncertainty estimates.
- The project has not been validated on live industrial sensor data.

## Future improvements

Useful next steps would be:

- evaluate with leave-one-run-out validation,
- compare against simple baselines such as linear regression and random forest,
- test additional time, frequency, and envelope features,
- evaluate whether the HI is monotonic and trendable across bearings,
- add model uncertainty or prediction intervals,
- version model artifacts together with training metrics,
- test the complete Docker image in CI.

## References

- NASA Prognostics Data Repository, IMS Bearings dataset: https://data.nasa.gov/dataset/ims-bearings
- H. Qiu, J. Lee, J. Lin, and G. Yu, "Wavelet filter-based weak signature detection method and its application on rolling element bearing prognostics," Journal of Sound and Vibration, 2006.
