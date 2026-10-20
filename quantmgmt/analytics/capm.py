import numpy as np
import pandas as pd


def capm_beta(
    asset_returns: pd.Series,
    market_returns: pd.Series,
) -> float:
    """
    Estimate CAPM beta of an asset relative to the market.

    Beta is estimated as:

        cov(R_i, R_m) / var(R_m)

    Parameters
    ----------
    asset_returns : pd.Series
        Asset returns.
    market_returns : pd.Series
        Market returns.

    Returns
    -------
    float
        Estimated beta.
    """
    data = pd.concat(
        [asset_returns, market_returns],
        axis=1,
    ).dropna()

    if len(data) < 2:
        raise ValueError(
            "at least two aligned observations are required"
        )

    asset = data.iloc[:, 0]
    market = data.iloc[:, 1]

    market_variance = market.var(ddof=1)

    if np.isclose(market_variance, 0.0):
        raise ValueError(
            "market variance must be positive"
        )

    covariance = asset.cov(market)

    return float(
        covariance / market_variance
    )


def estimate_sector_betas(
    returns: pd.DataFrame,
    market_returns: pd.Series,
) -> pd.Series:
    """
    Estimate CAPM beta for each asset column.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset return series.
    market_returns : pd.Series
        Market return series.

    Returns
    -------
    pd.Series
        CAPM betas indexed by asset.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(
            "returns must be a pandas DataFrame"
        )

    betas = {
        asset: capm_beta(
            returns[asset],
            market_returns,
        )
        for asset in returns.columns
    }

    return pd.Series(
        betas,
        name="beta",
        dtype=float,
    )


def estimate_sml(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """
    Estimate the empirical Security Market Line.

    Cross-sectional regression:

        E[R_i] = alpha + lambda * beta_i

    where beta_i is each asset's CAPM beta.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns.
    market_returns : pd.Series
        Market returns.
    periods_per_year : int, default 12
        Number of return observations per year.

    Returns
    -------
    pd.DataFrame
        DataFrame containing annualized mean return,
        CAPM beta, and fitted SML return for each asset.
    """
    if periods_per_year <= 0:
        raise ValueError(
            "periods_per_year must be positive"
        )

    betas = estimate_sector_betas(
        returns,
        market_returns,
    )

    annual_returns = (
        returns.mean(skipna=True)
        * periods_per_year
    )

    annual_returns = annual_returns.reindex(
        betas.index
    )

    valid = pd.concat(
        [
            betas.rename("beta"),
            annual_returns.rename(
                "annual_return"
            ),
        ],
        axis=1,
    ).dropna()

    if len(valid) < 2:
        raise ValueError(
            "at least two assets are required "
            "to estimate the SML"
        )

    x = valid["beta"].to_numpy()
    y = valid["annual_return"].to_numpy()

    design_matrix = np.column_stack(
        [
            np.ones(len(x)),
            x,
        ]
    )

    coefficients, _, _, _ = np.linalg.lstsq(
        design_matrix,
        y,
        rcond=None,
    )

    intercept = coefficients[0]
    slope = coefficients[1]

    fitted_returns = (
        intercept
        + slope * valid["beta"]
    )

    result = valid.copy()
    result["sml_fitted_return"] = fitted_returns

    result.attrs["intercept"] = float(
        intercept
    )
    result.attrs["slope"] = float(
        slope
    )

    return result