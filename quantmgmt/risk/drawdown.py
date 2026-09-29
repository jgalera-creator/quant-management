import pandas as pd
import numpy as np


def drawdown_series(
    returns: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Compute the drawdown series from periodic simple returns.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns expressed in decimal form.

    Returns
    -------
    pd.Series or pd.DataFrame
        Drawdown series, where 0 represents a previous peak
        and negative values represent losses from that peak.

    Raises
    ------
    TypeError
        If returns is not a pandas Series or DataFrame.

    Notes
    -----
    Wealth is calculated as:

        W_t = Product(1 + R_t)

    Drawdown is calculated as:

        DD_t = W_t / max(W_0, ..., W_t) - 1
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("returns must be a pandas Series or DataFrame")

    wealth = (1 + returns).cumprod()

    running_peak = wealth.cummax().clip(lower=1.0)

    return wealth / running_peak - 1


def max_drawdown(
    returns: pd.Series | pd.DataFrame,
) -> float | pd.Series:
    """
    Compute the maximum drawdown from periodic simple returns.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns expressed in decimal form.

    Returns
    -------
    float or pd.Series
        Maximum drawdown. For a DataFrame, returns one value per column.

    Raises
    ------
    TypeError
        If returns is not a pandas Series or DataFrame.
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("returns must be a pandas Series or DataFrame")

    drawdowns = drawdown_series(returns)

    return drawdowns.min()



def time_under_water(
    returns: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Compute consecutive periods spent below the previous wealth peak.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns expressed in decimal form.

    Returns
    -------
    pd.Series or pd.DataFrame
        Number of consecutive periods spent underwater.
        A value of 0 indicates that the portfolio is at a
        historical wealth peak.

    Raises
    ------
    TypeError
        If returns is not a pandas Series or DataFrame.

    Notes
    -----
    Time under water increases by one for every consecutive
    observation with a negative drawdown and resets to zero
    when the previous wealth peak is recovered.
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("returns must be a pandas Series or DataFrame")

    drawdowns = drawdown_series(returns)

    if isinstance(drawdowns, pd.Series):
        result = pd.Series(0, index=drawdowns.index, dtype=int)

        count = 0

        for idx, drawdown in drawdowns.items():
            if drawdown < 0:
                count += 1
            else:
                count = 0

            result.loc[idx] = count

        return result

    result = pd.DataFrame(
        0,
        index=drawdowns.index,
        columns=drawdowns.columns,
        dtype=int,
    )

    for column in drawdowns.columns:
        count = 0

        for idx, drawdown in drawdowns[column].items():
            if drawdown < 0:
                count += 1
            else:
                count = 0

            result.loc[idx, column] = count

    return result



def recovery_time(
    returns: pd.Series | pd.DataFrame,
) -> float | pd.Series:
    """
    Compute the recovery time of the maximum drawdown.

    Recovery time is defined as the number of periods from the
    trough of the maximum drawdown until the previous wealth peak
    is recovered.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns expressed in decimal form.

    Returns
    -------
    float or pd.Series
        Number of periods required to recover from the trough of
        the maximum drawdown.

        Returns NaN if the previous peak has not yet been recovered.

    Raises
    ------
    TypeError
        If returns is not a pandas Series or DataFrame.
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("returns must be a pandas Series or DataFrame")

    def _recovery_time_series(series: pd.Series) -> float:
        wealth = (1 + series).cumprod()

        running_peak = wealth.cummax().clip(lower=1.0)

        drawdowns = wealth / running_peak - 1

        trough_position = drawdowns.to_numpy().argmin()

        if np.isclose(drawdowns.iloc[trough_position], 0.0):
            return 0.0

        previous_peak = running_peak.iloc[trough_position]

        wealth_after_trough = wealth.iloc[trough_position + 1:]

        recovered = wealth_after_trough >= previous_peak

        if not recovered.any():
            return np.nan

        recovery_position = np.flatnonzero(recovered.to_numpy())[0]

        return float(recovery_position + 1)

    if isinstance(returns, pd.Series):
        return _recovery_time_series(returns)

    return returns.apply(_recovery_time_series)