"""Signal feature extraction used by training and inference."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pywt


DEFAULT_WAVELET = "morl"


def to_signal_array(signal: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert one vibration snapshot to a validated one-dimensional array."""
    array = np.asarray(signal, dtype=float)

    if array.ndim != 1:
        raise ValueError("Each vibration signal must be one-dimensional.")
    if array.size == 0:
        raise ValueError("Vibration signals cannot be empty.")
    if not np.isfinite(array).all():
        raise ValueError("Vibration signals must contain only finite numeric values.")

    return array


def calculate_health_indicator(
    signal: Sequence[float] | np.ndarray,
    wavelet_scale: float,
    wavelet_name: str = DEFAULT_WAVELET,
) -> float:
    """Calculate RMS of the real CWT coefficient at one selected scale."""
    if wavelet_scale <= 0:
        raise ValueError("wavelet_scale must be greater than zero.")

    array = to_signal_array(signal)
    coefficients, _ = pywt.cwt(array, scales=[wavelet_scale], wavelet=wavelet_name)
    filtered_signal = coefficients.real.ravel()
    return float(np.sqrt(np.mean(filtered_signal**2)))


def extract_health_indicators(
    signals: Sequence[Sequence[float] | np.ndarray],
    wavelet_scale: float,
    wavelet_name: str = DEFAULT_WAVELET,
) -> np.ndarray:
    """Calculate one health-indicator value for each vibration snapshot."""
    return np.asarray(
        [
            calculate_health_indicator(signal, wavelet_scale, wavelet_name)
            for signal in signals
        ],
        dtype=float,
    )
