import numpy as np
import pandas as pd

from quantmgmt.risk.drawdown import (
    max_drawdown,
    time_under_water,
)
from quantmgmt.risk.returns import cagr
from quantmgmt.risk.tail import cvar_historical



def capital_resilience_ratio(
    returns: pd.Series | pd.DataFrame,
    confidence_level: float = 0.95,
    periods_per_year: int = 252,
) -> float | pd.Series:
    """
    Compute the Capital Resilience Ratio (CRR).

    CRR measures compound growth relative to drawdown depth,
    tail-loss severity, and the persistence of underwater periods.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns expressed in decimal form.
    confidence_level : float, default=0.95
        Confidence level used for historical CVaR.
    periods_per_year : int, default=252
        Number of return periods in one year.

    Returns
    -------
    float or pd.Series
        Capital Resilience Ratio. Higher values indicate greater
        compound growth relative to downside risk and drawdown
        persistence.

    Raises
    ------
    TypeError
        If returns is not a pandas Series or DataFrame.
    ValueError
        If periods_per_year is not positive or confidence_level
        is not between 0 and 1.

    Notes
    -----
    CRR is defined as:

        CAGR
        ---------------------------------------------------
        sqrt(abs(Max Drawdown) * CVaR)
        * (1 + Max Time Under Water / periods_per_year)

    Max Time Under Water is normalized into years, allowing
    portfolios with different sampling frequencies to be compared
    when periods_per_year is specified correctly.
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("returns must be a pandas Series or DataFrame")

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    growth = cagr(
        returns,
        periods_per_year=periods_per_year,
    )

    drawdown = abs(max_drawdown(returns))

    tail_risk = cvar_historical(
        returns,
        confidence_level=confidence_level,
    )

    underwater = time_under_water(returns)

    max_underwater = underwater.max()

    underwater_years = (
        max_underwater / periods_per_year
    )

    risk_burden = (
        np.sqrt(drawdown * tail_risk)
        * (1 + underwater_years)
    )

    if isinstance(risk_burden, pd.Series):
        result = growth / risk_burden

        zero_risk = np.isclose(risk_burden, 0.0)

        result.loc[
            zero_risk & (growth > 0)
        ] = np.inf

        result.loc[
            zero_risk & np.isclose(growth, 0.0)
        ] = np.nan

        return result

    if np.isclose(risk_burden, 0.0):
        if growth > 0:
            return np.inf

        return np.nan

    return growth / risk_burden