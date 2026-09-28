"""Model loading and Remaining Useful Life inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from .features import extract_health_indicators


class BearingPrognosticator:
    """Predict RUL from a fixed window of raw vibration snapshots."""

    def __init__(self, model_path: str | Path, config: dict[str, Any]) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.window_size = int(config["window_size"])
        self.wavelet_scale = float(config.get("wavelet_scale", config.get("optimal_alpha", 0.0)))

        if self.window_size <= 0:
            raise ValueError("window_size must be greater than zero.")
        if self.wavelet_scale <= 0:
            raise ValueError("wavelet_scale must be greater than zero.")

        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)

    def predict_rul(self, raw_signal_sequence: list[list[float]]) -> float:
        """Predict RUL in dataset cycles for one chronological signal window."""
        if len(raw_signal_sequence) != self.window_size:
            raise ValueError(
                f"Expected {self.window_size} signal windows, received {len(raw_signal_sequence)}."
            )

        health_indicators = extract_health_indicators(
            raw_signal_sequence,
            wavelet_scale=self.wavelet_scale,
        )
        feature_vector = health_indicators.reshape(1, -1)
        prediction = float(self.model.predict(feature_vector)[0])

        if not np.isfinite(prediction):
            raise ValueError("Model returned a non-finite RUL prediction.")

        return prediction
