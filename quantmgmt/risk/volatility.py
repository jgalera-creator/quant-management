import numpy as np
import pandas as pd


def annualized_volatility(
    returns: pd.Series | pd.DataFrame,
    periods_per_year: int = 252,
) -> float | pd.Series:
    """
    Compute annualized volatility from periodic returns.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic returns expressed in decimal form.
    periods_per_year : int, default=252
        Number of return periods in one year.

    Returns
    -------
    float or pd.Series
        Annualized volatility.

    Raises
    ------
    TypeError
        If returns is not a pandas Series or DataFrame.
    ValueError
        If periods_per_year is not positive.

    Notes
    -----
    Annualized volatility is calculated as:

        std(R) * sqrt(periods_per_year)
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("returns must be a pandas Series or DataFrame")

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    return returns.std() * np.sqrt(periods_per_year)



def downside_deviation(
    returns: pd.Series | pd.DataFrame,
    target_return: float = 0.0,
    periods_per_year: int = 252,
) -> float | pd.Series:
    """
    Compute annualized downside deviation.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic returns expressed in decimal form.
    target_return : float, default=0.0
        Minimum acceptable return per period.
    periods_per_year : int, default=252
        Number of return periods in one year.

    Returns
    -------
    float or pd.Series
        Annualized downside deviation.

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

    downside = (returns - target_return).clip(upper=0)

    downside_variance = (downside ** 2).mean()

    return np.sqrt(downside_variance) * np.sqrt(periods_per_year)



def rolling_volatility(
    returns: pd.Series | pd.DataFrame,
    window: int = 20,
    periods_per_year: int = 252,
) -> pd.Series | pd.DataFrame:
    """
    Compute rolling annualized volatility.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic returns expressed in decimal form.
    window : int, default=20
        Number of observations in each rolling window.
    periods_per_year : int, default=252
        Number of return periods in one year.

    Returns
    -------
    pd.Series or pd.DataFrame
        Rolling annualized volatility.

    Raises
    ------
    TypeError
        If returns is not a pandas Series or DataFrame.
    ValueError
        If window or periods_per_year are not positive.
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("returns must be a pandas Series or DataFrame")

    if window <= 0:
        raise ValueError("window must be positive")

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    return (
        returns
        .rolling(window=window)
        .std()
        * np.sqrt(periods_per_year)
    )