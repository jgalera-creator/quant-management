import pandas as pd


def estimate_inputs(
    returns: pd.DataFrame,
    periods_per_year: int = 12,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Estimate expected returns and covariance matrix from historical returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns with observations in rows and assets in columns.
    periods_per_year : int, default=12
        Number of return periods per year.

    Returns
    -------
    expected_returns : pd.Series
        Annualized arithmetic expected returns.
    covariance_matrix : pd.DataFrame
        Annualized sample covariance matrix.

    Raises
    ------
    TypeError
        If returns is not a pandas DataFrame.
    ValueError
        If periods_per_year is not positive.
        If fewer than two complete observations are available.

    Notes
    -----
    Sample estimates are used:

        mu = mean(R) * periods_per_year

        Sigma = cov(R) * periods_per_year

    Rows containing missing values are removed so that all assets
    are estimated over the same observations.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    clean_returns = returns.dropna(how="any")

    if len(clean_returns) < 2:
        raise ValueError(
            "at least two complete observations are required"
        )

    expected_returns = (
        clean_returns.mean()
        * periods_per_year
    )

    covariance_matrix = (
        clean_returns.cov()
        * periods_per_year
    )

    return expected_returns, covariance_matrix