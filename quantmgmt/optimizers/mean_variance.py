import numpy as np
import pandas as pd
from scipy.optimize import minimize


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


def portfolio_return(
    weights: np.ndarray,
    expected_returns: pd.Series,
) -> float:
    """
    Compute expected portfolio return.
    """
    return float(
        weights @ expected_returns.to_numpy()
    )


def portfolio_variance(
    weights: np.ndarray,
    covariance_matrix: pd.DataFrame,
) -> float:
    """
    Compute portfolio variance.
    """
    covariance = covariance_matrix.to_numpy()

    return float(
        weights.T
        @ covariance
        @ weights
    )


def portfolio_volatility(
    weights: np.ndarray,
    covariance_matrix: pd.DataFrame,
) -> float:
    """
    Compute expected portfolio volatility.
    """
    variance = portfolio_variance(
        weights,
        covariance_matrix,
    )

    return float(
        np.sqrt(variance)
    )


def _build_bounds(
    n_assets: int,
    long_only: bool,
    max_weight: float | None = None,
) -> list[tuple[float | None, float | None]] | None:
    """
    Build bounds for portfolio optimization.
    """
    if max_weight is not None:
        if not 0 < max_weight <= 1:
            raise ValueError(
                "max_weight must be between 0 and 1"
            )

        if long_only and max_weight * n_assets < 1:
            raise ValueError(
                "max_weight is too restrictive "
                "for the number of assets"
            )

    if long_only:
        upper_bound = (
            max_weight
            if max_weight is not None
            else 1.0
        )

        return [
            (0.0, upper_bound)
            for _ in range(n_assets)
        ]

    if max_weight is not None:
        return [
            (-max_weight, max_weight)
            for _ in range(n_assets)
        ]

    return None


def min_variance(
    covariance_matrix: pd.DataFrame,
    long_only: bool = True,
    max_weight: float | None = None,
) -> pd.Series:
    """
    Compute the global minimum-variance portfolio.
    """
    n_assets = len(covariance_matrix)

    initial_weights = np.full(
        n_assets,
        1.0 / n_assets,
    )

    bounds = _build_bounds(
        n_assets=n_assets,
        long_only=long_only,
        max_weight=max_weight,
    )

    constraints = [
        {
            "type": "eq",
            "fun": lambda weights: (
                np.sum(weights) - 1.0
            ),
        }
    ]

    result = minimize(
        fun=portfolio_variance,
        x0=initial_weights,
        args=(covariance_matrix,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "ftol": 1e-12,
            "maxiter": 1000,
        },
    )

    if not result.success:
        raise ValueError(
            f"optimization failed: {result.message}"
        )

    return pd.Series(
        result.x,
        index=covariance_matrix.index,
        name="weight",
    )

def negative_sharpe_ratio(
    weights: np.ndarray,
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Compute negative Sharpe ratio for numerical minimization.
    """
    expected_return = portfolio_return(
        weights,
        expected_returns,
    )

    volatility = portfolio_volatility(
        weights,
        covariance_matrix,
    )

    if volatility <= 0:
        return np.inf

    sharpe = (
        expected_return - risk_free_rate
    ) / volatility

    return -sharpe


def max_sharpe(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
    long_only: bool = True,
    max_weight: float | None = None,
) -> pd.Series:
    """
    Compute the maximum-Sharpe portfolio.
    """
    if not expected_returns.index.equals(
        covariance_matrix.index
    ):
        raise ValueError(
            "expected_returns and covariance_matrix "
            "must contain assets in the same order"
        )

    n_assets = len(expected_returns)

    initial_weights = np.full(
        n_assets,
        1.0 / n_assets,
    )

    bounds = _build_bounds(
        n_assets=n_assets,
        long_only=long_only,
        max_weight=max_weight,
    )

    constraints = [
        {
            "type": "eq",
            "fun": lambda weights: (
                np.sum(weights) - 1.0
            ),
        }
    ]

    result = minimize(
        fun=negative_sharpe_ratio,
        x0=initial_weights,
        args=(
            expected_returns,
            covariance_matrix,
            risk_free_rate,
        ),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise ValueError(
            f"optimization failed: {result.message}"
        )

    return pd.Series(
        result.x,
        index=expected_returns.index,
        name="weight",
    )


def efficient_frontier(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    n_points: int = 50,
    long_only: bool = True,
    max_weight: float | None = None,
) -> pd.DataFrame:
    """
    Compute portfolios along the efficient frontier.

    For each target return, find the minimum-variance portfolio
    satisfying the portfolio constraints.
    """
    if not expected_returns.index.equals(
        covariance_matrix.index
    ):
        raise ValueError(
            "expected_returns and covariance_matrix "
            "must contain assets in the same order"
        )

    if n_points < 2:
        raise ValueError(
            "n_points must be at least 2"
        )

    n_assets = len(expected_returns)

    bounds = _build_bounds(
        n_assets=n_assets,
        long_only=long_only,
        max_weight=max_weight,
    )

    # Global minimum-variance portfolio
    min_var_weights = min_variance(
        covariance_matrix,
        long_only=long_only,
        max_weight=max_weight,
    )

    minimum_frontier_return = portfolio_return(
        min_var_weights.to_numpy(),
        expected_returns,
    )

    # Highest feasible expected-return portfolio
    max_return_result = minimize(
        fun=lambda weights: -portfolio_return(
            weights,
            expected_returns,
        ),
        x0=min_var_weights.to_numpy(),
        method="SLSQP",
        bounds=bounds,
        constraints=[
            {
                "type": "eq",
                "fun": lambda weights: (
                    np.sum(weights) - 1.0
                ),
            }
        ],
        options={
            "ftol": 1e-12,
            "maxiter": 1000,
        },
    )

    if not max_return_result.success:
        raise ValueError(
            "could not determine maximum feasible return"
        )

    maximum_frontier_return = portfolio_return(
        max_return_result.x,
        expected_returns,
    )

    target_returns = np.linspace(
        minimum_frontier_return,
        maximum_frontier_return,
        n_points,
    )

    frontier_rows = []

    # Start exactly at the global minimum-variance solution.
    first_row = {
        "return": minimum_frontier_return,
        "volatility": portfolio_volatility(
            min_var_weights.to_numpy(),
            covariance_matrix,
        ),
    }

    for asset, weight in min_var_weights.items():
        first_row[asset] = weight

    frontier_rows.append(first_row)

    # Warm start: each solution becomes the initial point
    # for the next target return.
    previous_weights = min_var_weights.to_numpy()

    for target_return in target_returns[1:]:
        constraints = [
            {
                "type": "eq",
                "fun": lambda weights: (
                    np.sum(weights) - 1.0
                ),
            },
            {
                "type": "eq",
                "fun": lambda weights, target=target_return: (
                    portfolio_return(
                        weights,
                        expected_returns,
                    )
                    - target
                ),
            },
        ]

        result = minimize(
            fun=portfolio_variance,
            x0=previous_weights,
            args=(covariance_matrix,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "ftol": 1e-12,
                "maxiter": 1000,
            },
        )

        if not result.success:
            raise ValueError(
                "efficient-frontier optimization failed "
                f"for target return {target_return:.6f}: "
                f"{result.message}"
            )

        weights = result.x

        row = {
            "return": portfolio_return(
                weights,
                expected_returns,
            ),
            "volatility": portfolio_volatility(
                weights,
                covariance_matrix,
            ),
        }

        for asset, weight in zip(
            expected_returns.index,
            weights,
        ):
            row[asset] = weight

        frontier_rows.append(row)

        previous_weights = weights

    return pd.DataFrame(frontier_rows)



def target_volatility(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    target_vol: float,
    long_only: bool = True,
    max_weight: float | None = None,
) -> pd.Series:
    """
    Find the highest-return portfolio for a target volatility.

    Parameters
    ----------
    expected_returns : pd.Series
        Annualized expected asset returns.
    covariance_matrix : pd.DataFrame
        Annualized covariance matrix.
    target_vol : float
        Target annualized portfolio volatility.
    long_only : bool, default=True
        If True, short positions are not allowed.
    max_weight : float or None, default=None
        Maximum weight allowed per asset.

    Returns
    -------
    pd.Series
        Optimal portfolio weights.

    Raises
    ------
    ValueError
        If inputs are inconsistent, the target volatility
        is invalid, or the optimization is infeasible.
    """
    if not expected_returns.index.equals(
        covariance_matrix.index
    ):
        raise ValueError(
            "expected_returns and covariance_matrix "
            "must contain assets in the same order"
        )

    if target_vol <= 0:
        raise ValueError(
            "target_vol must be positive"
        )

    n_assets = len(expected_returns)

    bounds = _build_bounds(
        n_assets=n_assets,
        long_only=long_only,
        max_weight=max_weight,
    )

    min_var_weights = min_variance(
        covariance_matrix,
        long_only=long_only,
        max_weight=max_weight,
    )

    min_volatility = portfolio_volatility(
        min_var_weights.to_numpy(),
        covariance_matrix,
    )

    if target_vol < min_volatility - 1e-8:
        raise ValueError(
            "target_vol is below the minimum feasible volatility"
        )

    constraints = [
        {
            "type": "eq",
            "fun": lambda weights: (
                np.sum(weights) - 1.0
            ),
        },
        {
            "type": "eq",
            "fun": lambda weights: (
                portfolio_variance(
                    weights,
                    covariance_matrix,
                )
                - target_vol**2
            ),
        },
    ]

    result = minimize(
        fun=lambda weights: -portfolio_return(
            weights,
            expected_returns,
        ),
        x0=min_var_weights.to_numpy(),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "ftol": 1e-12,
            "maxiter": 1000,
        },
    )

    if not result.success:
        raise ValueError(
            f"optimization failed: {result.message}"
        )

    achieved_volatility = portfolio_volatility(
        result.x,
        covariance_matrix,
    )

    if not np.isclose(
        achieved_volatility,
        target_vol,
        atol=1e-6,
    ):
        raise ValueError(
            "optimizer did not achieve the target volatility"
        )

    return pd.Series(
        result.x,
        index=expected_returns.index,
        name="weight",
    )