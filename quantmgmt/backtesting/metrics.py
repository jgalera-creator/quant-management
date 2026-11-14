import numpy as np
import pandas as pd

from quantmgmt.risk.returns import cagr
from quantmgmt.risk.volatility import annualized_volatility
from quantmgmt.risk.ratios import (
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
)
from quantmgmt.risk.drawdown import max_drawdown
from quantmgmt.risk.tail import (
    var_historical,
    cvar_historical,
)


def concentration_metrics(
    weights: pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute portfolio capital concentration.

    Returns
    -------
    pd.DataFrame
        HHI and effective number of assets.

    For a Series, returns one row.
    For a DataFrame, returns one row per observation.
    """
    if isinstance(weights, pd.Series):
        weights = weights.to_frame().T

    if not isinstance(weights, pd.DataFrame):
        raise TypeError(
            "weights must be a Series or DataFrame"
        )

    if weights.empty:
        raise ValueError("weights cannot be empty")

    values = weights.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("weights must be finite")

    if (values < -1e-10).any():
        raise ValueError("weights must be non-negative")

    if not np.allclose(
        values.sum(axis=1),
        1.0,
        atol=1e-8,
    ):
        raise ValueError("weights must sum to one")

    hhi = np.sum(values**2, axis=1)

    effective_assets = 1.0 / hhi

    return pd.DataFrame(
        {
            "hhi": hhi,
            "effective_assets": effective_assets,
        },
        index=weights.index,
    )



def annualized_turnover(
    turnover: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Annualize total traded-notional turnover.

    Turnover convention:
        absolute traded notional / pre-trade NAV

    Includes initial portfolio establishment.
    """
    if not isinstance(turnover, pd.Series):
        raise TypeError("turnover must be a Series")

    if turnover.empty:
        raise ValueError("turnover cannot be empty")

    if periods_per_year <= 0:
        raise ValueError(
            "periods_per_year must be positive"
        )

    values = turnover.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("turnover must be finite")

    if (values < 0).any():
        raise ValueError("turnover cannot be negative")

    return float(
        values.sum()
        * periods_per_year
        / len(values)
    )



def evaluate_backtest(
    comparison_result: dict,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    Evaluate a multi-strategy backtest.

    Parameters
    ----------
    comparison_result:
        Output of run_strategy_comparison().

    risk_free_rate:
        Annualized risk-free rate.

    periods_per_year:
        Trading periods per year.

    Returns
    -------
    pd.DataFrame
        One row per strategy and one column per metric.

    Notes
    -----
    Performance metrics use net portfolio returns.

    Concentration metrics use target allocations
    at rebalance dates.

    VaR, CVaR and maximum drawdown preserve the
    conventions of the existing risk modules.
    """
    if not isinstance(comparison_result, dict):
        raise TypeError(
            "comparison_result must be a dictionary"
        )

    required_keys = {
        "weights",
        "results",
        "nav",
        "portfolio_returns",
        "turnover",
        "costs",
    }

    if not required_keys.issubset(comparison_result):
        raise ValueError(
            "comparison_result is missing required keys"
        )

    if periods_per_year <= 0:
        raise ValueError(
            "periods_per_year must be positive"
        )

    if not np.isfinite(risk_free_rate):
        raise ValueError(
            "risk_free_rate must be finite"
        )

    portfolio_returns = comparison_result[
        "portfolio_returns"
    ]

    weights = comparison_result["weights"]
    turnover = comparison_result["turnover"]
    costs = comparison_result["costs"]

    if portfolio_returns.empty:
        raise ValueError(
            "portfolio returns cannot be empty"
        )

    if not np.isfinite(
        portfolio_returns.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "portfolio returns must be finite"
        )

    strategies = portfolio_returns.columns

    if (
        set(strategies) != set(weights)
        or set(strategies) != set(turnover.columns)
        or set(strategies) != set(costs.columns)
    ):
        raise ValueError(
            "strategy labels do not match"
        )

    rows = []

    for strategy in strategies:

        r = portfolio_returns[strategy]

        strategy_weights = weights[strategy]

        strategy_turnover = turnover[strategy]

        strategy_costs = costs[strategy]

        concentration = concentration_metrics(
            strategy_weights
        )

        row = {
            "strategy": strategy,

            "cagr": cagr(
                r,
                periods_per_year=periods_per_year,
            ),

            "volatility": annualized_volatility(
                r,
                periods_per_year=periods_per_year,
            ),

            "sharpe": sharpe_ratio(
                r,
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
            ),

            "sortino": sortino_ratio(
                r,
                target_return=0.0,
                periods_per_year=periods_per_year,
            ),

            "max_drawdown": max_drawdown(r),

            "calmar": calmar_ratio(
                r,
                periods_per_year=periods_per_year,
            ),

            "var_95": var_historical(
                r,
                confidence_level=0.95,
            ),

            "cvar_95": cvar_historical(
                r,
                confidence_level=0.95,
            ),

            "annual_turnover": annualized_turnover(
                strategy_turnover,
                periods_per_year=periods_per_year,
            ),

            "total_costs": float(
                strategy_costs.sum()
            ),

            "average_hhi": float(
                concentration["hhi"].mean()
            ),

            "average_effective_assets": float(
                concentration["effective_assets"].mean()
            ),
        }

        rows.append(row)

    return pd.DataFrame(
        rows
    ).set_index("strategy")