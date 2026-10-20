from __future__ import annotations

import pandas as pd
import statsmodels.api as sm


_FACTOR_MODELS = {
    "capm": [
        "Mkt-RF",
    ],
    "ff3": [
        "Mkt-RF",
        "SMB",
        "HML",
    ],
    "carhart4": [
        "Mkt-RF",
        "SMB",
        "HML",
        "Mom",
    ],
    "ff5": [
        "Mkt-RF",
        "SMB",
        "HML",
        "RMW",
        "CMA",
    ],
}


def run_factor_regression(
    returns: pd.Series,
    factors: pd.DataFrame,
    model: str,
    periods_per_year: int = 12,
) -> pd.Series:
    """
    Run a factor regression on excess asset returns.

    The regression is:

        R_i - RF = alpha + beta_1 F_1 + ... + epsilon

    Parameters
    ----------
    returns : pd.Series
        Asset or portfolio returns in decimal form.
    factors : pd.DataFrame
        Factor returns in decimal form. Must include RF.
    model : str
        One of:
        'capm', 'ff3', 'carhart4', 'ff5'.
    periods_per_year : int, default 12
        Used to annualize alpha.

    Returns
    -------
    pd.Series
        Annualized alpha, factor betas, t-statistics,
        adjusted R-squared, and number of observations.
    """
    model_key = model.lower()

    if model_key not in _FACTOR_MODELS:
        raise ValueError(
            f"unknown factor model: {model}. "
            f"Expected one of {list(_FACTOR_MODELS)}"
        )

    if periods_per_year <= 0:
        raise ValueError(
            "periods_per_year must be positive"
        )

    if not isinstance(
        returns,
        pd.Series,
    ):
        raise TypeError(
            "returns must be a pandas Series"
        )

    if not isinstance(
        factors,
        pd.DataFrame,
    ):
        raise TypeError(
            "factors must be a pandas DataFrame"
        )

    factor_names = _FACTOR_MODELS[
        model_key
    ]

    required_columns = (
        factor_names
        + ["RF"]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in factors.columns
    ]

    if missing_columns:
        raise ValueError(
            "missing factor columns: "
            + ", ".join(missing_columns)
        )

    data = pd.concat(
        [
            returns.rename("asset_return"),
            factors[required_columns],
        ],
        axis=1,
    ).dropna()

    minimum_observations = (
        len(factor_names) + 2
    )

    if len(data) < minimum_observations:
        raise ValueError(
            "not enough aligned observations "
            "for factor regression"
        )

    excess_returns = (
        data["asset_return"]
        - data["RF"]
    )

    x = data[factor_names]

    x = sm.add_constant(
        x,
        has_constant="add",
    )

    regression = sm.OLS(
        excess_returns,
        x,
    ).fit()

    result = {
        "alpha_annual": (
            regression.params["const"]
            * periods_per_year
        ),
        "alpha_tstat": (
            regression.tvalues["const"]
        ),
    }

    for factor in factor_names:
        result[
            f"beta_{factor}"
        ] = regression.params[factor]

        result[
            f"tstat_{factor}"
        ] = regression.tvalues[factor]

    result["adj_r_squared"] = (
        regression.rsquared_adj
    )

    result["n_obs"] = int(
        regression.nobs
    )

    return pd.Series(
        result,
        name=model_key,
    )



def rolling_factor_betas(
    returns: pd.Series,
    factors: pd.DataFrame,
    model: str,
    window: int = 36,
) -> pd.DataFrame:
    """
    Estimate rolling factor betas using fixed-length regressions.

    Parameters
    ----------
    returns : pd.Series
        Asset returns in decimal form.
    factors : pd.DataFrame
        Factor returns in decimal form, including RF.
    model : str
        One of:
        'capm', 'ff3', 'carhart4', 'ff5'.
    window : int, default 36
        Rolling window length in observations.

    Returns
    -------
    pd.DataFrame
        Rolling factor betas indexed by window end date.
    """
    model_key = model.lower()

    if model_key not in _FACTOR_MODELS:
        raise ValueError(
            f"unknown factor model: {model}. "
            f"Expected one of {list(_FACTOR_MODELS)}"
        )

    if window <= 0:
        raise ValueError(
            "window must be positive"
        )

    if not isinstance(
        returns,
        pd.Series,
    ):
        raise TypeError(
            "returns must be a pandas Series"
        )

    if not isinstance(
        factors,
        pd.DataFrame,
    ):
        raise TypeError(
            "factors must be a pandas DataFrame"
        )

    factor_names = _FACTOR_MODELS[
        model_key
    ]

    required_columns = (
        factor_names
        + ["RF"]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in factors.columns
    ]

    if missing_columns:
        raise ValueError(
            "missing factor columns: "
            + ", ".join(missing_columns)
        )

    data = pd.concat(
        [
            returns.rename("asset_return"),
            factors[required_columns],
        ],
        axis=1,
    ).dropna()

    if len(data) < window:
        raise ValueError(
            "not enough aligned observations "
            "for requested rolling window"
        )

    rows = []

    for end_position in range(
        window,
        len(data) + 1,
    ):
        window_data = data.iloc[
            end_position - window:end_position
        ]

        excess_returns = (
            window_data["asset_return"]
            - window_data["RF"]
        )

        x = window_data[
            factor_names
        ]

        x = sm.add_constant(
            x,
            has_constant="add",
        )

        regression = sm.OLS(
            excess_returns,
            x,
        ).fit()

        row = {
            f"beta_{factor}":
                regression.params[factor]
            for factor in factor_names
        }

        row["Date"] = (
            window_data.index[-1]
        )

        rows.append(row)

    result = pd.DataFrame(
        rows
    ).set_index("Date")

    return result