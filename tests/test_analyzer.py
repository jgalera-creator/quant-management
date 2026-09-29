import numpy as np
import pandas as pd
import pytest

from quantmgmt.risk.analyzer import RiskAnalyzer
from quantmgmt.risk.drawdown import max_drawdown
from quantmgmt.risk.returns import cagr
from quantmgmt.risk.volatility import annualized_volatility


@pytest.fixture
def sample_returns():
    return pd.Series([
        0.02,
        -0.01,
        0.03,
        -0.02,
        0.015,
    ])


def test_risk_analyzer_initialization(sample_returns):
    analyzer = RiskAnalyzer(
        sample_returns,
        periods_per_year=252,
    )

    assert analyzer.periods_per_year == 252

    pd.testing.assert_series_equal(
        analyzer.returns,
        sample_returns,
    )


def test_risk_analyzer_cagr(sample_returns):
    analyzer = RiskAnalyzer(
        sample_returns,
        periods_per_year=252,
    )

    expected = cagr(
        sample_returns,
        periods_per_year=252,
    )

    assert np.isclose(
        analyzer.cagr(),
        expected,
    )


def test_risk_analyzer_volatility(sample_returns):
    analyzer = RiskAnalyzer(
        sample_returns,
        periods_per_year=252,
    )

    expected = annualized_volatility(
        sample_returns,
        periods_per_year=252,
    )

    assert np.isclose(
        analyzer.volatility(),
        expected,
    )


def test_risk_analyzer_max_drawdown(sample_returns):
    analyzer = RiskAnalyzer(sample_returns)

    expected = max_drawdown(sample_returns)

    assert np.isclose(
        analyzer.max_drawdown(),
        expected,
    )


def test_risk_analyzer_invalid_input():
    with pytest.raises(TypeError):
        RiskAnalyzer(
            [0.01, -0.01]
        )


def test_risk_analyzer_invalid_periods(sample_returns):
    with pytest.raises(ValueError):
        RiskAnalyzer(
            sample_returns,
            periods_per_year=0,
        )


def test_risk_analyzer_invalid_var_method(sample_returns):
    analyzer = RiskAnalyzer(sample_returns)

    with pytest.raises(ValueError):
        analyzer.var(
            method="invalid",
        )


def test_risk_analyzer_summary_dataframe():
    returns = pd.DataFrame(
        {
            "Asset_A": [
                0.02,
                -0.01,
                0.03,
                -0.02,
                0.015,
            ],
            "Asset_B": [
                0.01,
                -0.005,
                0.015,
                -0.01,
                0.008,
            ],
        }
    )

    analyzer = RiskAnalyzer(
        returns,
        periods_per_year=5,
    )

    result = analyzer.summary(
        confidence_level=0.80,
    )

    assert isinstance(result, pd.DataFrame)

    assert list(result.index) == [
        "Asset_A",
        "Asset_B",
    ]

    expected_columns = {
        "Annualized Return",
        "CAGR",
        "Volatility",
        "Downside Deviation",
        "Sharpe",
        "Sortino",
        "Max Drawdown",
        "Recovery Time",
        "Skewness",
        "Kurtosis",
        "VaR Historical",
        "VaR Gaussian",
        "VaR Cornish-Fisher",
        "CVaR Historical",
        "CVaR Gaussian",
        "Calmar",
        "Capital Resilience",
    }

    assert expected_columns.issubset(
        result.columns
    )



def test_risk_analyzer_summary_series():
    returns = pd.Series(
        [
            0.02,
            -0.01,
            0.03,
            -0.02,
            0.015,
        ],
        name="Portfolio_A",
    )

    analyzer = RiskAnalyzer(
        returns,
        periods_per_year=5,
    )

    result = analyzer.summary(
        confidence_level=0.80,
    )

    assert isinstance(result, pd.DataFrame)

    assert result.shape[0] == 1

    assert result.index[0] == "Portfolio_A"



def test_summary_matches_individual_metrics():
    returns = pd.Series(
        [
            0.02,
            -0.01,
            0.03,
            -0.02,
            0.015,
        ],
        name="Portfolio_A",
    )

    analyzer = RiskAnalyzer(
        returns,
        periods_per_year=5,
    )

    summary = analyzer.summary(
        confidence_level=0.80,
    )

    assert np.isclose(
        summary.loc[
            "Portfolio_A",
            "Volatility",
        ],
        analyzer.volatility(),
    )

    assert np.isclose(
        summary.loc[
            "Portfolio_A",
            "Max Drawdown",
        ],
        analyzer.max_drawdown(),
    )

    assert np.isclose(
        summary.loc[
            "Portfolio_A",
            "CAGR",
        ],
        analyzer.cagr(),
    )