import numpy as np
import pandas as pd
import pytest

from quantmgmt.risk.custom import capital_resilience_ratio


def test_capital_resilience_ratio_series():
    returns = pd.Series([
        0.05,
        -0.03,
        0.04,
        -0.02,
        0.03,
        0.02,
    ])

    result = capital_resilience_ratio(
        returns,
        confidence_level=0.80,
        periods_per_year=6,
    )

    assert isinstance(result, float)
    assert np.isfinite(result)



def test_capital_resilience_ratio_dataframe():
    returns = pd.DataFrame(
        {
            "Portfolio_A": [
                0.05,
                -0.03,
                0.04,
                -0.02,
                0.03,
                0.02,
            ],
            "Portfolio_B": [
                0.03,
                -0.01,
                0.02,
                -0.01,
                0.02,
                0.01,
            ],
        }
    )

    result = capital_resilience_ratio(
        returns,
        confidence_level=0.80,
        periods_per_year=6,
    )

    assert isinstance(result, pd.Series)

    assert "Portfolio_A" in result.index
    assert "Portfolio_B" in result.index



def test_crr_penalizes_longer_underwater_periods():
    fast_recovery = pd.Series([
        0.10,
        -0.10,
        0.12,
        0.02,
        0.02,
        0.02,
    ])

    slow_recovery = pd.Series([
        0.10,
        -0.10,
        0.02,
        0.02,
        0.02,
        0.08,
    ])

    fast_result = capital_resilience_ratio(
        fast_recovery,
        confidence_level=0.80,
        periods_per_year=6,
    )

    slow_result = capital_resilience_ratio(
        slow_recovery,
        confidence_level=0.80,
        periods_per_year=6,
    )

    assert fast_result > slow_result



def test_crr_can_be_negative():
    returns = pd.Series([
        -0.05,
        -0.04,
        0.01,
        -0.03,
        0.01,
    ])

    result = capital_resilience_ratio(
        returns,
        confidence_level=0.80,
        periods_per_year=5,
    )

    assert result < 0



def test_crr_positive_growth_without_observed_risk_is_infinite():
    returns = pd.Series([
        0.01,
        0.02,
        0.01,
        0.02,
    ])

    result = capital_resilience_ratio(
        returns,
        confidence_level=0.80,
        periods_per_year=4,
    )

    assert np.isinf(result)


def test_crr_invalid_input():
    with pytest.raises(TypeError):
        capital_resilience_ratio(
            [0.01, -0.01]
        )


def test_crr_invalid_periods():
    returns = pd.Series([
        0.01,
        -0.01,
    ])

    with pytest.raises(ValueError):
        capital_resilience_ratio(
            returns,
            periods_per_year=0,
        )


def test_crr_invalid_confidence_level():
    returns = pd.Series([
        0.01,
        -0.01,
    ])

    with pytest.raises(ValueError):
        capital_resilience_ratio(
            returns,
            confidence_level=1.0,
        )