import numpy as np
import pytest

from scripts.train_model import create_sliding_window_data


def test_sliding_window_targets_count_down_to_failure():
    hi = np.array([10, 11, 12, 13, 14], dtype=float)

    features, targets = create_sliding_window_data(hi, window_size=2)

    assert features.tolist() == [[10, 11], [11, 12], [12, 13]]
    assert targets.tolist() == [2.0, 1.0, 0.0]


def test_sliding_window_requires_enough_history():
    with pytest.raises(ValueError, match="Not enough"):
        create_sliding_window_data(np.array([1.0, 2.0]), window_size=2)
