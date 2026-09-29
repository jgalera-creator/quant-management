import numpy as np
import pandas as pd

from quantmgmt.risk.volatility import downside_deviation
from quantmgmt.risk.drawdown import max_drawdown
from quantmgmt.risk.returns import cagr

def sharpe_ratio(
    returns: pd.Series | pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float | pd.Series:
    """
    Compute the annualized Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns expressed in decimal form.
    risk_free_rate : float, default=0.0
        Annual risk-free rate expressed in decimal form.
        For example, 0.03 represents 3% per year.
    periods_per_year : int, default=252
        Number of return periods in one year.

    Returns
    -------
    float or pd.Series
        Annualized Sharpe ratio.

    Raises
    ------
    TypeError
        If returns is not a pandas Series or DataFrame.
    ValueError
        If periods_per_year is not positive.
        If risk_free_rate is less than or equal to -100%.

    Notes
    -----
    The annual risk-free rate is converted to an equivalent
    periodic rate using compound interest:

        rf_period = (1 + rf_annual) ** (1 / periods_per_year) - 1

    The annualized Sharpe ratio is:

        mean(excess_returns) / std(excess_returns)
        * sqrt(periods_per_year)
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("returns must be a pandas Series or DataFrame")

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    if risk_free_rate <= -1:
        raise ValueError("risk_free_rate must be greater than -1")

    risk_free_per_period = (
        (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    )

    excess_returns = returns - risk_free_per_period

    volatility = excess_returns.std()

    return (
        excess_returns.mean()
        / volatility
        * np.sqrt(periods_per_year)
    )


def sortino_ratio(
    returns: pd.Series | pd.DataFrame,
    target_return: float = 0.0,
    periods_per_year: int = 252,
) -> float | pd.Series:
    """
    Compute the annualized Sortino ratio.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns expressed in decimal form.
    target_return : float, default=0.0
        Annual minimum acceptable return expressed in decimal form.
        For example, 0.03 represents 3% per year.
    periods_per_year : int, default=252
        Number of return periods in one year.

    Returns
    -------
    float or pd.Series
        Annualized Sortino ratio.

    Raises
    ------
    TypeError
        If returns is not a pandas Series or DataFrame.
    ValueError
        If periods_per_year is not positive.
        If target_return is less than or equal to -100%.

    Notes
    -----
    The annual target return is converted to an equivalent
    periodic target:

        target_period =
            (1 + target_return) ** (1 / periods_per_year) - 1

    Sortino ratio is calculated as:

        annualized excess return / annualized downside deviation
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("returns must be a pandas Series or DataFrame")

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    if target_return <= -1:
        raise ValueError("target_return must be greater than -1")

    target_per_period = (
        (1 + target_return) ** (1 / periods_per_year) - 1
    )

    excess_returns = returns - target_per_period

    annualized_excess_return = (
        excess_returns.mean() * periods_per_year
    )

    downside_risk = downside_deviation(
        returns,
        target_return=target_per_period,
        periods_per_year=periods_per_year,
    )

    return annualized_excess_return / downside_risk



def calmar_ratio(
    returns: pd.Series | pd.DataFrame,
    periods_per_year: int = 252,
) -> float | pd.Series:
    """
    Compute the Calmar ratio.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns expressed in decimal form.
    periods_per_year : int, default=252
        Number of return periods in one year.

    Returns
    -------
    float or pd.Series
        CAGR divided by the absolute value of maximum drawdown.

    Raises
    ------
    TypeError
        If returns is not a pandas Series or DataFrame.
    ValueError
        If periods_per_year is not positive.

    Notes
    -----
    Calmar ratio is calculated as:

        CAGR / abs(Max Drawdown)

    If maximum drawdown is zero and CAGR is positive,
    the ratio is positive infinity.
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("returns must be a pandas Series or DataFrame")

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    annual_growth = cagr(
        returns,
        periods_per_year=periods_per_year,
    )

    maximum_drawdown = abs(max_drawdown(returns))

    if isinstance(annual_growth, pd.Series):
        result = annual_growth / maximum_drawdown

        zero_drawdown = np.isclose(maximum_drawdown, 0.0)

        result.loc[
            zero_drawdown & (annual_growth > 0)
        ] = np.inf

        result.loc[
            zero_drawdown & np.isclose(annual_growth, 0.0)
        ] = np.nan

        return result

    if np.isclose(maximum_drawdown, 0.0):
        if annual_growth > 0:
            return np.inf

        return np.nan

    return annual_growth / maximum_drawdown


def tracking_error(
    returns: pd.Series | pd.DataFrame,
    benchmark: pd.Series | pd.DataFrame,
    periods_per_year: int = 252,
) -> float | pd.Series:
    """
    Compute annualized tracking error relative to a benchmark.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Portfolio or asset periodic returns.
    benchmark : pd.Series or pd.DataFrame
        Benchmark periodic returns.
    periods_per_year : int, default=252
        Number of return periods in one year.

    Returns
    -------
    float or pd.Series
        Annualized tracking error.

    Raises
    ------
    TypeError
        If returns or benchmark is not a pandas Series or DataFrame.
    ValueError
        If periods_per_year is not positive.

    Notes
    -----
    Tracking error is calculated as:

        std(returns - benchmark) * sqrt(periods_per_year)

    Pandas aligns inputs by index before subtraction.
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("returns must be a pandas Series or DataFrame")

    if not isinstance(benchmark, (pd.Series, pd.DataFrame)):
        raise TypeError("benchmark must be a pandas Series or DataFrame")

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    active_returns = returns.subtract(
        benchmark,
        axis=0,
    )

    return active_returns.std() * np.sqrt(periods_per_year)