import numpy as np
import pandas as pd
import pytest

from quantmgmt.risk.volatility import (
    annualized_volatility,
    downside_deviation,
    rolling_volatility,
)


def test_annualized_volatility_series():
    returns = pd.Series([0.01, -0.02, 0.015, -0.005])

    result = annualized_volatility(
        returns,
        periods_per_year=252,
    )

    expected = returns.std() * np.sqrt(252)

    assert np.isclose(result, expected)


def test_annualized_volatility_dataframe():
    returns = pd.DataFrame(
        {
            "Asset_A": [0.01, -0.02, 0.015, -0.005],
            "Asset_B": [0.005, -0.005, 0.005, -0.005],
        }
    )

    result = annualized_volatility(
        returns,
        periods_per_year=252,
    )

    expected = returns.std() * np.sqrt(252)

    pd.testing.assert_series_equal(result, expected)


def test_annualized_volatility_zero_returns():
    returns = pd.Series([0.0, 0.0, 0.0, 0.0])

    result = annualized_volatility(returns)

    assert np.isclose(result, 0.0)


def test_annualized_volatility_invalid_periods():
    returns = pd.Series([0.01, -0.01])

    with pytest.raises(ValueError):
        annualized_volatility(
            returns,
            periods_per_year=0,
        )


def test_annualized_volatility_invalid_input():
    with pytest.raises(TypeError):
        annualized_volatility([0.01, -0.01])



def test_downside_deviation_series():
    returns = pd.Series([0.02, -0.01, 0.03, -0.02])

    result = downside_deviation(
        returns,
        target_return=0.0,
        periods_per_year=4,
    )

    downside = pd.Series([0.0, -0.01, 0.0, -0.02])
    expected = np.sqrt((downside ** 2).mean()) * np.sqrt(4)

    assert np.isclose(result, expected)


def test_downside_deviation_no_downside():
    returns = pd.Series([0.01, 0.02, 0.03])

    result = downside_deviation(returns)

    assert np.isclose(result, 0.0)


def test_downside_deviation_dataframe():
    returns = pd.DataFrame(
        {
            "Asset_A": [0.02, -0.01, 0.03, -0.02],
            "Asset_B": [0.01, 0.02, 0.01, 0.03],
        }
    )

    result = downside_deviation(
        returns,
        periods_per_year=4,
    )

    assert isinstance(result, pd.Series)
    assert result["Asset_A"] > 0
    assert np.isclose(result["Asset_B"], 0.0)


def test_downside_deviation_invalid_periods():
    returns = pd.Series([0.01, -0.01])

    with pytest.raises(ValueError):
        downside_deviation(
            returns,
            periods_per_year=0,
        )



def test_rolling_volatility_series():
    returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02])

    result = rolling_volatility(
        returns,
        window=3,
        periods_per_year=252,
    )

    expected = (
        returns
        .rolling(window=3)
        .std()
        * np.sqrt(252)
    )

    pd.testing.assert_series_equal(result, expected)


def test_rolling_volatility_dataframe():
    returns = pd.DataFrame(
        {
            "Asset_A": [0.01, -0.02, 0.015, -0.005],
            "Asset_B": [0.005, -0.005, 0.01, -0.01],
        }
    )

    result = rolling_volatility(
        returns,
        window=2,
        periods_per_year=252,
    )

    expected = (
        returns
        .rolling(window=2)
        .std()
        * np.sqrt(252)
    )

    pd.testing.assert_frame_equal(result, expected)


def test_rolling_volatility_initial_values_are_nan():
    returns = pd.Series([0.01, -0.02, 0.015])

    result = rolling_volatility(
        returns,
        window=3,
    )

    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[1])
    assert not np.isnan(result.iloc[2])


def test_rolling_volatility_invalid_window():
    returns = pd.Series([0.01, -0.01])

    with pytest.raises(ValueError):
        rolling_volatility(
            returns,
            window=0,
        )