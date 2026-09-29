import numpy as np
import pandas as pd
import pytest

from quantmgmt.risk.ratios import (
    calmar_ratio,
    sharpe_ratio,
    sortino_ratio,
    tracking_error,
)

def test_sharpe_ratio_series():
    returns = pd.Series([
        0.01,
        -0.005,
        0.015,
        0.002,
    ])

    result = sharpe_ratio(
        returns,
        risk_free_rate=0.0,
        periods_per_year=252,
    )

    expected = (
        returns.mean()
        / returns.std()
        * np.sqrt(252)
    )

    assert np.isclose(result, expected)



def test_sharpe_ratio_with_risk_free_rate():
    returns = pd.Series([
        0.01,
        0.015,
        -0.005,
        0.02,
    ])

    risk_free_rate = 0.03
    periods_per_year = 252

    rf_period = (
        (1 + risk_free_rate)
        ** (1 / periods_per_year)
        - 1
    )

    excess_returns = returns - rf_period

    expected = (
        excess_returns.mean()
        / excess_returns.std()
        * np.sqrt(periods_per_year)
    )

    result = sharpe_ratio(
        returns,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )

    assert np.isclose(result, expected)



def test_sharpe_ratio_dataframe():
    returns = pd.DataFrame(
        {
            "Asset_A": [
                0.01,
                -0.005,
                0.015,
                0.002,
            ],
            "Asset_B": [
                0.005,
                0.004,
                -0.003,
                0.006,
            ],
        }
    )

    result = sharpe_ratio(
        returns,
        periods_per_year=252,
    )

    expected = (
        returns.mean()
        / returns.std()
        * np.sqrt(252)
    )

    pd.testing.assert_series_equal(
        result,
        expected,
    )


def test_sharpe_ratio_invalid_periods():
    returns = pd.Series([0.01, -0.01])

    with pytest.raises(ValueError):
        sharpe_ratio(
            returns,
            periods_per_year=0,
        )


def test_sharpe_ratio_invalid_input():
    with pytest.raises(TypeError):
        sharpe_ratio([0.01, -0.01])


def test_sharpe_ratio_invalid_risk_free_rate():
    returns = pd.Series([0.01, -0.01])

    with pytest.raises(ValueError):
        sharpe_ratio(
            returns,
            risk_free_rate=-1.0,
        )



def test_sortino_ratio_series():
    returns = pd.Series([
        0.02,
        -0.01,
        0.03,
        -0.02,
    ])

    periods_per_year = 12

    result = sortino_ratio(
        returns,
        target_return=0.0,
        periods_per_year=periods_per_year,
    )

    downside = returns.clip(upper=0)

    downside_risk = (
        np.sqrt((downside ** 2).mean())
        * np.sqrt(periods_per_year)
    )

    annualized_return = (
        returns.mean() * periods_per_year
    )

    expected = annualized_return / downside_risk

    assert np.isclose(result, expected)


def test_sortino_ratio_with_target_return():
    returns = pd.Series([
        0.01,
        0.015,
        -0.005,
        0.02,
    ])

    target_return = 0.03
    periods_per_year = 252

    target_period = (
        (1 + target_return)
        ** (1 / periods_per_year)
        - 1
    )

    excess_returns = returns - target_period

    downside = excess_returns.clip(upper=0)

    downside_risk = (
        np.sqrt((downside ** 2).mean())
        * np.sqrt(periods_per_year)
    )

    expected = (
        excess_returns.mean()
        * periods_per_year
        / downside_risk
    )

    result = sortino_ratio(
        returns,
        target_return=target_return,
        periods_per_year=periods_per_year,
    )

    assert np.isclose(result, expected)



def test_sortino_ratio_dataframe():
    returns = pd.DataFrame(
        {
            "Asset_A": [
                0.02,
                -0.01,
                0.03,
                -0.02,
            ],
            "Asset_B": [
                0.01,
                -0.005,
                0.015,
                -0.002,
            ],
        }
    )

    result = sortino_ratio(
        returns,
        periods_per_year=12,
    )

    assert isinstance(result, pd.Series)
    assert "Asset_A" in result.index
    assert "Asset_B" in result.index



def test_sortino_ratio_invalid_periods():
    returns = pd.Series([0.01, -0.01])

    with pytest.raises(ValueError):
        sortino_ratio(
            returns,
            periods_per_year=0,
        )


def test_sortino_ratio_invalid_input():
    with pytest.raises(TypeError):
        sortino_ratio([0.01, -0.01])


def test_sortino_ratio_invalid_target_return():
    returns = pd.Series([0.01, -0.01])

    with pytest.raises(ValueError):
        sortino_ratio(
            returns,
            target_return=-1.0,
        )




def test_calmar_ratio_series():
    returns = pd.Series([
        0.10,
        -0.20,
        0.10,
        0.15,
    ])

    periods_per_year = 4

    result = calmar_ratio(
        returns,
        periods_per_year=periods_per_year,
    )

    total_growth = (1 + returns).prod()

    expected_cagr = (
        total_growth ** (periods_per_year / len(returns))
        - 1
    )

    wealth = (1 + returns).cumprod()
    running_peak = wealth.cummax().clip(lower=1.0)
    drawdowns = wealth / running_peak - 1

    expected_mdd = abs(drawdowns.min())

    expected = expected_cagr / expected_mdd

    assert np.isclose(result, expected)



def test_calmar_ratio_dataframe():
    returns = pd.DataFrame(
        {
            "Asset_A": [
                0.10,
                -0.20,
                0.10,
                0.15,
            ],
            "Asset_B": [
                0.05,
                -0.05,
                0.04,
                0.03,
            ],
        }
    )

    result = calmar_ratio(
        returns,
        periods_per_year=4,
    )

    assert isinstance(result, pd.Series)
    assert "Asset_A" in result.index
    assert "Asset_B" in result.index


def test_calmar_ratio_no_drawdown():
    returns = pd.Series([
        0.02,
        0.03,
        0.01,
    ])

    result = calmar_ratio(
        returns,
        periods_per_year=3,
    )

    assert np.isinf(result)



def test_calmar_ratio_invalid_periods():
    returns = pd.Series([0.01, -0.01])

    with pytest.raises(ValueError):
        calmar_ratio(
            returns,
            periods_per_year=0,
        )


def test_calmar_ratio_invalid_input():
    with pytest.raises(TypeError):
        calmar_ratio([0.01, -0.01])



def test_tracking_error_series():
    returns = pd.Series([
        0.01,
        -0.005,
        0.015,
        0.002,
    ])

    benchmark = pd.Series([
        0.008,
        -0.004,
        0.010,
        0.001,
    ])

    periods_per_year = 252

    active_returns = returns - benchmark

    expected = (
        active_returns.std()
        * np.sqrt(periods_per_year)
    )

    result = tracking_error(
        returns,
        benchmark,
        periods_per_year=periods_per_year,
    )

    assert np.isclose(result, expected)



def test_tracking_error_dataframe():
    returns = pd.DataFrame(
        {
            "Portfolio_A": [
                0.01,
                -0.005,
                0.015,
                0.002,
            ],
            "Portfolio_B": [
                0.012,
                -0.003,
                0.009,
                0.004,
            ],
        }
    )

    benchmark = pd.Series([
        0.008,
        -0.004,
        0.010,
        0.001,
    ])

    result = tracking_error(
        returns,
        benchmark,
        periods_per_year=252,
    )

    assert isinstance(result, pd.Series)

    expected_a = (
        (returns["Portfolio_A"] - benchmark).std()
        * np.sqrt(252)
    )

    expected_b = (
        (returns["Portfolio_B"] - benchmark).std()
        * np.sqrt(252)
    )

    assert np.isclose(
        result["Portfolio_A"],
        expected_a,
    )

    assert np.isclose(
        result["Portfolio_B"],
        expected_b,
    )


def test_tracking_error_identical_to_benchmark():
    returns = pd.Series([
        0.01,
        -0.005,
        0.015,
    ])

    result = tracking_error(
        returns,
        returns.copy(),
    )

    assert np.isclose(result, 0.0)


def test_tracking_error_invalid_periods():
    returns = pd.Series([0.01, -0.01])
    benchmark = pd.Series([0.005, -0.005])

    with pytest.raises(ValueError):
        tracking_error(
            returns,
            benchmark,
            periods_per_year=0,
        )


def test_tracking_error_invalid_returns():
    benchmark = pd.Series([0.005, -0.005])

    with pytest.raises(TypeError):
        tracking_error(
            [0.01, -0.01],
            benchmark,
        )


def test_tracking_error_invalid_benchmark():
    returns = pd.Series([0.01, -0.01])

    with pytest.raises(TypeError):
        tracking_error(
            returns,
            [0.005, -0.005],
        )