import numpy as np
import pytest

import bearing_app.prognosticator as prognosticator_module
from bearing_app.prognosticator import BearingPrognosticator


class DummyModel:
    def predict(self, features):
        assert features.shape == (1, 3)
        return np.array([12.5])


def make_predictor():
    predictor = object.__new__(BearingPrognosticator)
    predictor.window_size = 3
    predictor.wavelet_scale = 6.0
    predictor.model = DummyModel()
    return predictor


def test_predict_rul_uses_one_health_indicator_per_signal(monkeypatch):
    predictor = make_predictor()
    monkeypatch.setattr(
        prognosticator_module,
        "extract_health_indicators",
        lambda signals, wavelet_scale: np.array([1.0, 2.0, 3.0]),
    )

    result = predictor.predict_rul([[1.0], [2.0], [3.0]])

    assert result == 12.5


def test_predict_rul_rejects_wrong_window_size():
    predictor = make_predictor()

    with pytest.raises(ValueError, match="Expected 3"):
        predictor.predict_rul([[1.0], [2.0]])
