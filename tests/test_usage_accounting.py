import pytest

from usage_accounting.usage_accounting import calculate_cost


def test_success_calculate_cost():
    cost = calculate_cost(
        input_tokens=100,
        output_tokens=50,
        input_price=0.03,
        output_price=0.17,
    )

    expected_cost = (
        (100 / 1_000_000) * 0.03
        + (50 / 1_000_000) * 0.17
    )

    assert cost == pytest.approx(expected_cost)


def test_failure_negative_tokens():
    with pytest.raises(ValueError):
        calculate_cost(
            input_tokens=-100,
            output_tokens=50,
            input_price=0.03,
            output_price=0.17,
        )