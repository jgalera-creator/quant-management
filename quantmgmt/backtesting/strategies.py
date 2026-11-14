import numpy as np
import pandas as pd

from quantmgmt.optimizers.mean_variance import (
    estimate_inputs,
    min_variance,
    max_sharpe,
)

from quantmgmt.optimizers.black_litterman import (
    implied_equilibrium_returns,
    posterior_returns,
    build_omega,
)

from quantmgmt.optimizers.hrp import hrp_weights
from quantmgmt.optimizers.mean_variance import OptimizationError


def compute_strategy_weights(
    returns: pd.DataFrame,
    strategy: str,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    max_weight: float | None = None,
    market_weights: pd.Series | None = None,
    risk_aversion: float | None = None,
    P: pd.DataFrame | None = None,
    Q: pd.Series | None = None,
    confidences: pd.Series | None = None,
    tau: float = 0.05,
) -> pd.Series:
    """
    Compute target portfolio weights for one strategy.

    Strategies:
        equal_weight
        min_variance
        max_sharpe
        black_litterman
        hrp

    All estimation inputs must contain historical
    information available at the decision date.

    BL convention:
        pi and Q represent annualized excess returns.
        The BL optimizer therefore uses risk_free_rate=0.
    """

    supported = {
        "equal_weight",
        "min_variance",
        "max_sharpe",
        "black_litterman",
        "hrp",
    }

    if strategy not in supported:
        raise ValueError(
            f"unsupported strategy: {strategy}"
        )

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a DataFrame")

    if returns.empty:
        raise ValueError("returns cannot be empty")

    if not returns.columns.is_unique:
        raise ValueError("asset labels must be unique")

    if returns.isna().any().any():
        raise ValueError("returns contain missing values")

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("returns must be finite")

    if len(returns) < 2:
        raise ValueError(
            "at least two observations are required"
        )

    if periods_per_year <= 0:
        raise ValueError(
            "periods_per_year must be positive"
        )

    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite")

    assets = returns.columns
    n_assets = len(assets)

    if max_weight is not None:
        if not np.isfinite(max_weight) or not 0 < max_weight <= 1:
            raise ValueError("invalid max_weight")

        if max_weight * n_assets < 1 - 1e-10:
            raise ValueError(
                "max_weight is infeasible"
            )

        if strategy == "hrp":
            raise ValueError(
                "HRP does not support max_weight constraints"
            )

    # Equal-weight benchmark
    if strategy == "equal_weight":

        weights = pd.Series(
            1.0 / n_assets,
            index=assets,
            name="weight",
        )

    else:

        expected_returns, covariance = estimate_inputs(
            returns,
            periods_per_year=periods_per_year,
        )

        if strategy == "min_variance":

            weights = min_variance(
                covariance_matrix=covariance,
                long_only=True,
                max_weight=max_weight,
            )

        elif strategy == "max_sharpe":

            weights = max_sharpe(
                expected_returns=expected_returns,
                covariance_matrix=covariance,
                risk_free_rate=risk_free_rate,
                long_only=True,
                max_weight=max_weight,
            )

        elif strategy == "hrp":

            weights = hrp_weights(
                covariance_matrix=covariance,
            )

        elif strategy == "black_litterman":

            if any(
                value is None
                for value in [
                    market_weights,
                    risk_aversion,
                    P,
                    Q,
                ]
            ):
                raise ValueError(
                    "Black-Litterman requires "
                    "market_weights, risk_aversion, P and Q"
                )

            # 1. Equilibrium excess returns
            pi = implied_equilibrium_returns(
                covariance_matrix=covariance,
                market_weights=market_weights,
                risk_aversion=risk_aversion,
            )

            # 2. View uncertainty
            omega = build_omega(
                covariance_matrix=covariance,
                P=P,
                tau=tau,
                confidences=confidences,
            )

            # 3. Posterior excess returns
            posterior = posterior_returns(
                equilibrium_returns=pi,
                covariance_matrix=covariance,
                P=P,
                Q=Q,
                omega=omega,
                tau=tau,
            )

            # 4. Portfolio optimization
            # Posterior returns are excess returns,
            # therefore risk_free_rate=0.
            weights = max_sharpe(
                expected_returns=posterior,
                covariance_matrix=covariance,
                risk_free_rate=0.0,
                long_only=True,
                max_weight=max_weight,
            )

    # Common output validation
    weights = weights.reindex(assets)

    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError("non-finite strategy weights")

    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError("weights do not sum to one")

    if (weights < -1e-8).any():
        raise ValueError("negative weights")

    if max_weight is not None and (weights > max_weight + 1e-6).any():
        raise ValueError("weights exceed max_weight")

    return weights.rename("weight")





class StrategyOptimizationError(RuntimeError):
    """A portfolio optimizer failed to find a valid solution."""