import pandas as pd
from statistics import NormalDist
from typing import Literal


def skewness(
    returns: pd.Series | pd.DataFrame,
) -> float | pd.Series:
    """
    Compute sample skewness of periodic returns.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic returns expressed in decimal form.

    Returns
    -------
    float or pd.Series
        Sample skewness of the return distribution.

    Raises
    ------
    TypeError
        If returns is not a pandas Series or DataFrame.

    Notes
    -----
    Negative skewness indicates a distribution with a heavier
    or longer left tail, while positive skewness indicates
    asymmetry toward the right tail.
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("returns must be a pandas Series or DataFrame")

    return returns.skew()



def kurtosis(
    returns: pd.Series | pd.DataFrame,
) -> float | pd.Series:
    """
    Compute sample excess kurtosis of periodic returns.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic returns expressed in decimal form.

    Returns
    -------
    float or pd.Series
        Sample excess kurtosis of the return distribution.

    Raises
    ------
    TypeError
        If returns is not a pandas Series or DataFrame.

    Notes
    -----
    Pandas returns Fisher's excess kurtosis:

        Normal distribution -> kurtosis approximately 0

    Positive values indicate heavier tails than a normal
    distribution.
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("returns must be a pandas Series or DataFrame")

    return returns.kurt()


def var_historical(
    returns: pd.Series | pd.DataFrame,
    confidence_level: float = 0.95,
) -> float | pd.Series:
    """
    Compute historical Value at Risk (VaR).

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic returns expressed in decimal form.
    confidence_level : float, default=0.95
        Confidence level for VaR.
        For example, 0.95 represents 95%.

    Returns
    -------
    float or pd.Series
        Historical VaR expressed as a positive loss magnitude.

    Raises
    ------
    TypeError
        If returns is not a pandas Series or DataFrame.
    ValueError
        If confidence_level is not between 0 and 1.

    Notes
    -----
    Historical VaR is calculated as:

        VaR = -quantile(returns, 1 - confidence_level)

    A positive result represents a loss magnitude.
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("returns must be a pandas Series or DataFrame")

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    quantile_level = round(1 - confidence_level, 12)

    quantile = returns.quantile(quantile_level)

    if isinstance(quantile, pd.Series):
        return (-quantile).clip(lower=0.0)

    return max(0.0, -quantile)



def var_parametric(
    returns: pd.Series | pd.DataFrame,
    confidence_level: float = 0.95,
    method: Literal["gaussian", "cornish-fisher"] = "gaussian",
) -> float | pd.Series:
    """
    Compute parametric Value at Risk (VaR).

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic returns expressed in decimal form.
    confidence_level : float, default=0.95
        Confidence level for VaR.
    method : {"gaussian", "cornish-fisher"}, default="gaussian"
        Parametric VaR method.

        - "gaussian": assumes normally distributed returns.
        - "cornish-fisher": adjusts the normal quantile using
          skewness and excess kurtosis.

    Returns
    -------
    float or pd.Series
        Parametric VaR expressed as a positive loss magnitude.

    Raises
    ------
    TypeError
        If returns is not a pandas Series or DataFrame.
    ValueError
        If confidence_level is not between 0 and 1.
        If method is invalid.

    Notes
    -----
    Gaussian VaR:

        q = mean + z_alpha * std

    Cornish-Fisher adjustment:

        z_cf =
            z
            + (z^2 - 1) * S / 6
            + (z^3 - 3z) * K / 24
            - (2z^3 - 5z) * S^2 / 36

    where S is skewness and K is excess kurtosis.
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("returns must be a pandas Series or DataFrame")

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    if method not in {"gaussian", "cornish-fisher"}:
        raise ValueError(
            "method must be either 'gaussian' or 'cornish-fisher'"
        )

    alpha = 1 - confidence_level

    z_score = NormalDist().inv_cdf(alpha)

    if method == "gaussian":
        adjusted_z = z_score

    else:
        skew = skewness(returns)
        kurt = kurtosis(returns)

        adjusted_z = (
            z_score
            + ((z_score ** 2 - 1) * skew / 6)
            + ((z_score ** 3 - 3 * z_score) * kurt / 24)
            - (
                (2 * z_score ** 3 - 5 * z_score)
                * (skew ** 2)
                / 36
            )
        )

    quantile = (
        returns.mean()
        + adjusted_z * returns.std()
    )

    if isinstance(quantile, pd.Series):
        return (-quantile).clip(lower=0.0)

    return max(0.0, -quantile)



def cvar_historical(
    returns: pd.Series | pd.DataFrame,
    confidence_level: float = 0.95,
) -> float | pd.Series:
    """
    Compute historical Conditional Value at Risk (CVaR).

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic returns expressed in decimal form.
    confidence_level : float, default=0.95
        Confidence level for CVaR.
        For example, 0.95 represents 95%.

    Returns
    -------
    float or pd.Series
        Historical CVaR expressed as a positive loss magnitude.

    Raises
    ------
    TypeError
        If returns is not a pandas Series or DataFrame.
    ValueError
        If confidence_level is not between 0 and 1.

    Notes
    -----
    CVaR, also known as Expected Shortfall, is the average return
    in the tail beyond the historical VaR threshold:

        CVaR = -E[R | R <= q_alpha]

    where:

        alpha = 1 - confidence_level

    The result is returned as a positive loss magnitude.
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("returns must be a pandas Series or DataFrame")

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    alpha = round(1 - confidence_level, 12)

    if isinstance(returns, pd.Series):
        threshold = returns.quantile(alpha)

        tail_returns = returns[
            returns <= threshold
        ]

        return max(
            0.0,
            -tail_returns.mean(),
        )

    def _column_cvar(series: pd.Series) -> float:
        threshold = series.quantile(alpha)

        tail_returns = series[
            series <= threshold
        ]

        return max(
            0.0,
            -tail_returns.mean(),
        )

    return returns.apply(_column_cvar)




def cvar_parametric(
    returns: pd.Series | pd.DataFrame,
    confidence_level: float = 0.95,
) -> float | pd.Series:
    """
    Compute Gaussian parametric Conditional Value at Risk (CVaR).

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic returns expressed in decimal form.
    confidence_level : float, default=0.95
        Confidence level for CVaR.

    Returns
    -------
    float or pd.Series
        Parametric CVaR expressed as a positive loss magnitude.

    Raises
    ------
    TypeError
        If returns is not a pandas Series or DataFrame.
    ValueError
        If confidence_level is not between 0 and 1.

    Notes
    -----
    Returns are assumed to follow a normal distribution.

    Let:

        alpha = 1 - confidence_level
        z_alpha = Phi^(-1)(alpha)

    Gaussian Expected Shortfall is:

        ES = -mean + std * phi(z_alpha) / alpha

    where phi is the standard normal probability density function.

    The result is returned as a positive loss magnitude.
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("returns must be a pandas Series or DataFrame")

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    alpha = 1 - confidence_level

    normal = NormalDist()

    z_score = normal.inv_cdf(alpha)

    tail_density = normal.pdf(z_score)

    expected_shortfall = (
        -returns.mean()
        + returns.std() * tail_density / alpha
    )

    if isinstance(expected_shortfall, pd.Series):
        return expected_shortfall.clip(lower=0.0)

    return max(0.0, expected_shortfall)