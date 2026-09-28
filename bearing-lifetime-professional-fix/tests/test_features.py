import numpy as np
import pytest

from bearing_app.features import calculate_health_indicator, to_signal_array


def test_health_indicator_is_finite_and_non_negative():
    signal = np.sin(np.linspace(0, 8 * np.pi, 512))
    value = calculate_health_indicator(signal, wavelet_scale=6.0)

    assert np.isfinite(value)
    assert value >= 0


def test_signal_validation_rejects_empty_input():
    with pytest.raises(ValueError, match="cannot be empty"):
        to_signal_array([])


def test_signal_validation_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        to_signal_array([0.1, np.nan, 0.2])
