import numpy as np

from ecpp.evaluation.metrics import count_zero_crossings, first_threshold_time


def test_zero_crossing_count_uses_deadband():
    values = np.array([0.1, 0.0, -0.1, 0.1, 0.001])
    assert count_zero_crossings(values, deadband=0.01) == 2


def test_first_threshold_time_returns_first_ratio_hit():
    values = np.array([10.0, 5.0, 1.0, 0.5])
    times = np.array([0.0, 1.0, 2.0, 3.0])
    assert first_threshold_time(values, times, 0.1) == 2.0
