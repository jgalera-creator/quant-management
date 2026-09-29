import numpy as np
import pandas as pd
import pytest

from quantmgmt.risk.returns import (
    annualized_return,
    cagr,
    cumulative_returns,
    to_returns,
)


def test_to_returns_simple_series():
    prices = pd.Series(
        [100.0, 105.0, 103.0],
        index=pd.date_range("2026-01-01", periods=3),
    )

    result = to_returns(prices, method="simple")

    expected = pd.Series(
        [
            105 / 100 - 1,
            103 / 105 - 1,
        ],
        index=prices.index[1:],
    )

    pd.testing.assert_series_equal(result, expected)


def test_to_returns_log_series():
    prices = pd.Series(
        [100.0, 105.0, 103.0],
        index=pd.date_range("2026-01-01", periods=3),
    )

    result = to_returns(prices, method="log")

    expected = pd.Series(
        [
            np.log(105 / 100),
            np.log(103 / 105),
        ],
        index=prices.index[1:],
    )

    pd.testing.assert_series_equal(result, expected)


def test_to_returns_dataframe():
    prices = pd.DataFrame(
        {
            "Asset_A": [100.0, 110.0, 121.0],
            "Asset_B": [200.0, 180.0, 189.0],
        }
    )

    result = to_returns(prices)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)

    assert np.isclose(result.iloc[0]["Asset_A"], 0.10)
    assert np.isclose(result.iloc[0]["Asset_B"], -0.10)


def test_to_returns_invalid_method():
    prices = pd.Series([100.0, 101.0])

    with pytest.raises(ValueError):
        to_returns(prices, method="invalid")


def test_to_returns_invalid_input():
    with pytest.raises(TypeError):
        to_returns([100.0, 101.0, 102.0])


def test_log_returns_reject_non_positive_prices():
    prices = pd.Series([100.0, 0.0, 105.0])

    with pytest.raises(ValueError):
        to_returns(prices, method="log")



def test_cumulative_returns_series():
    returns = pd.Series([0.10, -0.10])

    result = cumulative_returns(returns)

    expected = pd.Series([
        0.10,
        -0.01,
    ])

    pd.testing.assert_series_equal(result, expected)


def test_cumulative_returns_dataframe():
    returns = pd.DataFrame(
        {
            "Asset_A": [0.10, -0.10],
            "Asset_B": [0.05, 0.05],
        }
    )

    result = cumulative_returns(returns)

    assert np.isclose(result.iloc[-1]["Asset_A"], -0.01)
    assert np.isclose(result.iloc[-1]["Asset_B"], 0.1025)


def test_cumulative_returns_invalid_input():
    with pytest.raises(TypeError):
        cumulative_returns([0.01, 0.02])



def test_annualized_return_series():
    returns = pd.Series([0.01, 0.02, -0.01])

    result = annualized_return(
        returns,
        periods_per_year=12,
    )

    expected = returns.mean() * 12

    assert np.isclose(result, expected)


def test_annualized_return_dataframe():
    returns = pd.DataFrame(
        {
            "Asset_A": [0.01, 0.02, 0.03],
            "Asset_B": [-0.01, 0.00, 0.01],
        }
    )

    result = annualized_return(
        returns,
        periods_per_year=12,
    )

    assert isinstance(result, pd.Series)
    assert np.isclose(result["Asset_A"], 0.24)
    assert np.isclose(result["Asset_B"], 0.00)


def test_cagr_series():
    returns = pd.Series([0.10, -0.10])

    result = cagr(
        returns,
        periods_per_year=2,
    )

    expected = (1.10 * 0.90) ** (2 / 2) - 1

    assert np.isclose(result, expected)


def test_cagr_dataframe():
    returns = pd.DataFrame(
        {
            "Asset_A": [0.10, -0.10],
            "Asset_B": [0.05, 0.05],
        }
    )

    result = cagr(
        returns,
        periods_per_year=2,
    )

    expected_a = (1.10 * 0.90) - 1
    expected_b = (1.05 * 1.05) - 1

    assert np.isclose(result["Asset_A"], expected_a)
    assert np.isclose(result["Asset_B"], expected_b)


def test_annualized_return_invalid_periods():
    returns = pd.Series([0.01, 0.02])

    with pytest.raises(ValueError):
        annualized_return(returns, periods_per_year=0)


def test_cagr_invalid_periods():
    returns = pd.Series([0.01, 0.02])

    with pytest.raises(ValueError):
        cagr(returns, periods_per_year=0)