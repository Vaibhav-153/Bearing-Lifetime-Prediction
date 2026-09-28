"""Train and evaluate the bearing RUL model from IMS vibration files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from bearing_app.features import extract_health_indicators, to_signal_array

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ASSET_DIR = ROOT / "bearing_app" / "Assets"
REPORT_DIR = ROOT / "reports"

WINDOW_SIZE = 15
SCALE_CANDIDATES = np.linspace(1.0, 10.0, 20)
RANDOM_STATE = 42

TRAIN_RUNS = (("2nd_test", 0), ("3rd_test", 2))
VALIDATION_RUN = ("1st_test", 7)


def load_signal_column(run_name: str, signal_column: int) -> list[np.ndarray]:
    """Load one signal column from all files in a run, sorted chronologically."""
    run_dir = DATA_DIR / run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {run_dir}")

    files = sorted(path for path in run_dir.iterdir() if path.is_file())
    if not files:
        raise ValueError(f"No data files found in {run_dir}")

    signals: list[np.ndarray] = []
    for path in files:
        frame = pd.read_csv(path, sep="\t", header=None)
        if signal_column >= frame.shape[1]:
            raise ValueError(
                f"Column {signal_column} does not exist in {path.name}; "
                f"file has {frame.shape[1]} columns."
            )
        signals.append(to_signal_array(frame.iloc[:, signal_column].to_numpy()))

    return signals


def periodicity_score(signal: np.ndarray, wavelet_scale: float) -> float:
    """Score one CWT scale using the leading singular-value ratio."""
    coefficients, _ = pywt.cwt(signal, scales=[wavelet_scale], wavelet="morl")
    envelope = np.abs(coefficients.ravel())
    usable_size = (envelope.size // 100) * 100
    if usable_size < 200:
        raise ValueError("Signal is too short for the periodicity calculation.")

    matrix = envelope[:usable_size].reshape(-1, 100)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size < 2 or singular_values[1] == 0:
        return float(singular_values[0])
    return float(singular_values[0] / singular_values[1])


def select_wavelet_scale(signal: np.ndarray) -> float:
    """Select the CWT scale with the highest periodicity score."""
    scores = {float(scale): periodicity_score(signal, float(scale)) for scale in SCALE_CANDIDATES}
    return max(scores, key=scores.get)


def create_sliding_window_data(
    health_indicators: np.ndarray,
    window_size: int = WINDOW_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Create feature windows and countdown RUL labels."""
    health_indicators = np.asarray(health_indicators, dtype=float)
    if health_indicators.ndim != 1:
        raise ValueError("health_indicators must be one-dimensional.")
    if health_indicators.size <= window_size:
        raise ValueError("Not enough health-indicator values for the requested window size.")

    rul = np.arange(health_indicators.size - 1, -1, -1, dtype=float)
    features = []
    targets = []

    for start in range(health_indicators.size - window_size):
        stop = start + window_size
        features.append(health_indicators[start:stop])
        targets.append(rul[stop])

    return np.asarray(features), np.asarray(targets)


def build_model() -> xgb.XGBRegressor:
    """Return the XGBoost configuration used by the original project."""
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def main() -> None:
    first_training_signals = load_signal_column(*TRAIN_RUNS[0])
    tuning_signal = first_training_signals[-100]
    wavelet_scale = select_wavelet_scale(tuning_signal)

    train_features = []
    train_targets = []

    for index, (run_name, signal_column) in enumerate(TRAIN_RUNS):
        signals = (
            first_training_signals if index == 0 else load_signal_column(run_name, signal_column)
        )
        health_indicators = extract_health_indicators(signals, wavelet_scale)
        features, targets = create_sliding_window_data(health_indicators)
        train_features.append(features)
        train_targets.append(targets)

    x_train = np.vstack(train_features)
    y_train = np.concatenate(train_targets)

    model = build_model()
    model.fit(x_train, y_train)

    validation_signals = load_signal_column(*VALIDATION_RUN)
    validation_hi = extract_health_indicators(validation_signals, wavelet_scale)
    x_validation, y_validation = create_sliding_window_data(validation_hi)
    predictions = model.predict(x_validation)

    rmse = float(np.sqrt(mean_squared_error(y_validation, predictions)))
    mae = float(mean_absolute_error(y_validation, predictions))

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model.save_model(ASSET_DIR / "model.json")
    with (ASSET_DIR / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump(
            {"wavelet_scale": wavelet_scale, "window_size": WINDOW_SIZE},
            config_file,
            indent=2,
        )

    metrics = {
        "validation_run": VALIDATION_RUN[0],
        "validation_signal_column": VALIDATION_RUN[1],
        "rmse_cycles": rmse,
        "mae_cycles": mae,
        "wavelet_scale": wavelet_scale,
        "window_size": WINDOW_SIZE,
        "random_state": RANDOM_STATE,
    }
    with (REPORT_DIR / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print(f"Selected wavelet scale: {wavelet_scale:.6f}")
    print(f"Validation RMSE: {rmse:.2f} cycles")
    print(f"Validation MAE: {mae:.2f} cycles")
    print(f"Model saved to: {ASSET_DIR / 'model.json'}")


if __name__ == "__main__":
    main()
