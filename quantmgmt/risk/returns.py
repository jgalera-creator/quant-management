from typing import Literal

import numpy as np
import pandas as pd


ReturnType = Literal["simple", "log"]


def to_returns(
    prices: pd.Series | pd.DataFrame,
    method: ReturnType = "simple",
) -> pd.Series | pd.DataFrame:
    """
    Convert a price series into returns.

    Parameters
    ----------
    prices : pd.Series or pd.DataFrame
        Price observations indexed by date.
    method : {"simple", "log"}, default="simple"
        Return calculation method.

        - "simple":
            R_t = P_t / P_{t-1} - 1

        - "log":
            r_t = ln(P_t / P_{t-1})

    Returns
    -------
    pd.Series or pd.DataFrame
        Returns with the first observation removed because no previous
        price is available.

    Raises
    ------
    TypeError
        If prices is not a pandas Series or DataFrame.
    ValueError
        If method is not "simple" or "log".
        If log returns are requested and prices contain non-positive values.
    """
    if not isinstance(prices, (pd.Series, pd.DataFrame)):
        raise TypeError("prices must be a pandas Series or DataFrame")

    if method not in {"simple", "log"}:
        raise ValueError("method must be either 'simple' or 'log'")

    if method == "simple":
        return prices.pct_change(fill_method=None).dropna()

    if (prices <= 0).any().any() if isinstance(prices, pd.DataFrame) else (prices <= 0).any():
        raise ValueError("log returns require strictly positive prices")

    return np.log(prices / prices.shift(1)).dropna()




def cumulative_returns(
    returns: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Compute cumulative returns from simple periodic returns.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Simple periodic returns expressed in decimal form.

    Returns
    -------
    pd.Series or pd.DataFrame
        Cumulative return series.

    Raises
    ------
    TypeError
        If returns is not a pandas Series or DataFrame.

    Notes
    -----
    Cumulative simple returns are calculated as:

        (1 + R_1)(1 + R_2)...(1 + R_t) - 1

    This function expects simple returns, not logarithmic returns.
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("returns must be a pandas Series or DataFrame")

    return (1 + returns).cumprod() - 1



def annualized_return(
    returns: pd.Series | pd.DataFrame,
    periods_per_year: int = 252,
) -> float | pd.Series:
    """
    Compute the arithmetic annualized return.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns.
    periods_per_year : int, default=252
        Number of return periods in one year.

    Returns
    -------
    float or pd.Series
        Arithmetic mean return annualized by periods_per_year.

    Raises
    ------
    TypeError
        If returns is not a pandas Series or DataFrame.
    ValueError
        If periods_per_year is not positive.
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("returns must be a pandas Series or DataFrame")

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    return returns.mean() * periods_per_year




def cagr(
    returns: pd.Series | pd.DataFrame,
    periods_per_year: int = 252,
) -> float | pd.Series:
    """
    Compute the Compound Annual Growth Rate (CAGR).

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns.
    periods_per_year : int, default=252
        Number of return periods in one year.

    Returns
    -------
    float or pd.Series
        Compound annual growth rate.

    Raises
    ------
    TypeError
        If returns is not a pandas Series or DataFrame.
    ValueError
        If periods_per_year is not positive.

    Notes
    -----
    CAGR is calculated as:

        (Product(1 + R_t)) ** (periods_per_year / n_periods) - 1
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("returns must be a pandas Series or DataFrame")

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    n_periods = returns.count()

    total_growth = (1 + returns).prod()

    return total_growth ** (periods_per_year / n_periods) - 1