import pytest

from parameter_sweep.parameter_sweep import measure_variance


def test_success_measure_variance():
    outputs = ["A", "A", "B", "C"]

    unique_outputs, variance_rate = measure_variance(outputs)

    assert unique_outputs == 3
    assert variance_rate == 0.75


def test_failure_measure_variance_empty_outputs():
    with pytest.raises(ValueError):
        measure_variance([])