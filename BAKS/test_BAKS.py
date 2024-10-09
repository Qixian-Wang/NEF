import numpy as np
import pytest

from BAKS import BAKSFiringRate, spike_detection
from BAKS import _bayesian_adaptive_kernel_smoother


def test_bayesian_adaptive_kernel_smoother_function():
    spike_data = np.array([0.1, 0.5, 1.0, 1.5])
    shape_parameter = 1
    scale_parameter_coefficient = len(spike_data) ** 0.8
    total_duration = spike_data[-1] - spike_data[0]
    spike_times_local = spike_data - spike_data[0]

    firing_rate, bandwidth = _bayesian_adaptive_kernel_smoother(spike_times_local, total_duration, shape_parameter, scale_parameter_coefficient)

    assert firing_rate > 0
    assert bandwidth > 0

def test_call_function_and_firing_list():
    baks_firing_rate = BAKSFiringRate()

    spike_data = [np.array([0.1, 0.5, 1.0, 1.5]), np.array([0.2, 0.6, 1.1, 1.6])]
    baks_firing_rate(spike_data)

    assert len(baks_firing_rate.firing_rate_list) == 2
    assert baks_firing_rate.firing_rate_list[0] != baks_firing_rate.firing_rate_list[1]

    for ch, firing_rate, rate in baks_firing_rate.firing_rate_list:
        assert firing_rate > 0, "firing_rate should be positive"
        assert rate >0, "Metric should be positive"

def test_call_metric_calculation():
    baks_firing_rate = BAKSFiringRate()
    input = np.array([0.1, 0.5, 1.0, 1.5])
    spike_data = [input]
    baks_firing_rate(spike_data)

    for ch, firing_rate, rate in baks_firing_rate.firing_rate_list:
        assert rate == len(input) / (input[-1]-input[0]), "Metric should be positive"

def test_call_zero_input():
    baks_firing_rate = BAKSFiringRate()

    spike_data = []
    baks_firing_rate(spike_data)

    for ch, firing_rate, rate in baks_firing_rate.firing_rate_list:
        assert firing_rate == 0, "FiringRate should be positive"
        assert rate == 0, "Metric should be positive"

def test_BAKS_with_different_data_length():
    baks_firing_rate = BAKSFiringRate()

    spike_data = [np.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]), np.array([0.2, 0.6, 1.1, 1.6])]
    baks_firing_rate(spike_data)

    assert len(baks_firing_rate.firing_rate_list) > 0

    for ch, firing_rate, rate in baks_firing_rate.firing_rate_list:
        assert firing_rate > 0, "FiringRate should be positive"

if __name__ == '__main__':
    pytest.main()