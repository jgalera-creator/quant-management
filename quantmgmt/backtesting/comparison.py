import numpy as np
import pandas as pd

from quantmgmt.backtesting.engine import (
    simulate_portfolio,
)

from quantmgmt.backtesting.walk_forward import (
    generate_walk_forward_with_fallback,
)

from quantmgmt.backtesting.strategies import (
    compute_strategy_weights,
)

from quantmgmt.backtesting.views import (
    momentum_relative_view,
)


STRATEGIES = (
    "equal_weight",
    "min_variance",
    "max_sharpe",
    "black_litterman",
    "hrp",
)


def default_market_weights() -> pd.Series:
    """
    Hypothetical strategic benchmark for the 12-ETF universe.

    These weights are fixed ex ante and do not represent
    historical market-capitalization weights.
    """
    return pd.Series(
        {
            "SPY": 0.22,
            "IWM": 0.06,
            "EFA": 0.12,
            "EEM": 0.08,
            "TLT": 0.10,
            "IEF": 0.10,
            "LQD": 0.07,
            "HYG": 0.04,
            "GLD": 0.08,
            "DBC": 0.04,
            "VNQ": 0.04,
            "TIP": 0.05,
        },
        name="market_weight",
    )


def run_strategy_comparison(
    returns: pd.DataFrame,
    market_weights: pd.Series,
    risk_free_rate: float = 0.0,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    momentum_months: int = 12,
    expected_spread: float = 0.03,
    confidence: float = 0.60,
    lookback_months: int = 36,
    initial_capital: float = 100_000.0,
    transaction_cost_bps: float = 0.0,
) -> dict:
    """
    Run a common walk-forward comparison of five strategies.

    All strategies:
        - use the same return observations
        - use the same rebalance dates
        - use the same historical estimation windows
        - execute through the same simulator

    Black-Litterman views are generated exclusively
    from information available at each decision date.

    Parameters
    ----------
    returns:
        Daily simple asset returns.

    market_weights:
        Fixed strategic benchmark weights, indexed by asset.

    risk_free_rate:
        Annualized risk-free rate used for maximum Sharpe.
        The baseline uses zero.

    Returns
    -------
    dict:
        weights:
            Dictionary of target-weight DataFrames.

        results:
            Dictionary of simulation results.

        nav:
            DataFrame containing portfolio NAVs.

        portfolio_returns:
            DataFrame containing daily portfolio returns.

        turnover:
            DataFrame containing turnover.

        costs:
            DataFrame containing transaction costs.
    """

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a DataFrame")

    if returns.empty:
        raise ValueError("returns cannot be empty")

    if not isinstance(market_weights, pd.Series):
        raise TypeError("market_weights must be a Series")

    if (
        not returns.columns.is_unique
        or not market_weights.index.is_unique
    ):
        raise ValueError("asset labels must be unique")

    if set(returns.columns) != set(market_weights.index):
        raise ValueError(
            "market weights and return assets do not match"
        )

    benchmark = market_weights.reindex(
        returns.columns
    ).astype(float)

    if not np.isfinite(benchmark.to_numpy()).all():
        raise ValueError("market weights must be finite")

    if (benchmark < 0).any():
        raise ValueError(
            "market weights must be non-negative"
        )

    if not np.isclose(benchmark.sum(), 1.0):
        raise ValueError(
            "market weights must sum to one"
        )

    all_weights = {}
    all_results = {}
    all_incidents = {}

    for strategy in STRATEGIES:

        def weight_function(
            historical_returns: pd.DataFrame,
            rebalance_date: pd.Timestamp,
            strategy_name: str = strategy,
        ) -> pd.Series:

            if strategy_name == "black_litterman":

                P, Q, confidences = momentum_relative_view(
                    historical_returns=historical_returns,
                    rebalance_date=rebalance_date,
                    momentum_months=momentum_months,
                    expected_spread=expected_spread,
                    confidence=confidence,
                )

                return compute_strategy_weights(
                    returns=historical_returns,
                    strategy="black_litterman",
                    market_weights=benchmark,
                    risk_aversion=risk_aversion,
                    P=P,
                    Q=Q,
                    confidences=confidences,
                    tau=tau,
                )

            return compute_strategy_weights(
                returns=historical_returns,
                strategy=strategy_name,
                risk_free_rate=risk_free_rate,
            )

        # Generate historical portfolio decisions.
        target_weights, incidents = generate_walk_forward_with_fallback(
            returns=returns,
            weight_function=weight_function,
            lookback_months=lookback_months,
        )

        if target_weights.empty:
            raise ValueError(
                "insufficient history for walk-forward"
            )

        # Simulate portfolio evolution.
        simulation = simulate_portfolio(
            returns=returns,
            target_weights=target_weights,
            initial_capital=initial_capital,
            transaction_cost_bps=transaction_cost_bps,
        )

        all_weights[strategy] = target_weights
        all_results[strategy] = simulation
        all_incidents[strategy] = incidents

    # Build comparable time series.
    nav = pd.DataFrame({
        strategy: result["nav"]
        for strategy, result in all_results.items()
    })

    portfolio_returns = pd.DataFrame({
        strategy: result["portfolio_returns"]
        for strategy, result in all_results.items()
    })

    turnover = pd.DataFrame({
        strategy: result["turnover"]
        for strategy, result in all_results.items()
    })

    costs = pd.DataFrame({
        strategy: result["costs"]
        for strategy, result in all_results.items()
    })

    # All strategies must have identical timestamps.
    if (
        nav.isna().any().any()
        or not nav.index.is_monotonic_increasing
    ):
        raise ValueError(
            "strategy NAV series are not aligned"
        )

    if nav.empty:
        raise ValueError(
            "comparison produced no observations"
        )

    return {
        "weights": all_weights,
        "results": all_results,
        "nav": nav,
        "portfolio_returns": portfolio_returns,
        "turnover": turnover,
        "costs": costs,
        "incidents": all_incidents,
    }