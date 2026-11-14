import pandas as pd
import numpy as np
from scipy.optimize import brentq


def quarterly_rebalance_dates(
    dates: pd.DatetimeIndex,
    lookback_months: int = 36,
) -> pd.DatetimeIndex:
    """
    Identify quarter-end rebalance dates with sufficient history.

    Only completed quarters are included.

    A rebalance decision at date t uses information
    available through t. The resulting weights become
    effective for subsequent market returns.
    """
    if not isinstance(dates, pd.DatetimeIndex):
        raise TypeError("dates must be a DatetimeIndex")

    if not dates.is_monotonic_increasing:
        raise ValueError("dates must be sorted")

    if not dates.is_unique:
        raise ValueError("dates must be unique")

    if dates.hasnans:
        raise ValueError("dates cannot contain NaT")

    if lookback_months <= 0:
        raise ValueError("lookback_months must be positive")

    if dates.empty:
        return pd.DatetimeIndex([])

    quarters = dates.to_period("Q")

    rebalance_dates = []

    for quarter in quarters.unique():

        quarter_dates = dates[quarters == quarter]

        last_date = quarter_dates[-1]

        # Approximate final business day of the quarter.
        quarter_end = quarter.end_time.normalize()

        expected_last_business_day = pd.bdate_range(
            end=quarter_end,
            periods=1,
        )[-1]

        # Reject quarters whose last observation is too
        # far from the expected quarter-end session.
        earliest_acceptable_date = (
            expected_last_business_day
            - pd.offsets.BDay(3)
        )

        if last_date < earliest_acceptable_date:
            continue

        # A following observation in the next quarter
        # is required to avoid including an unfinished
        # final quarter.
        future_dates = dates[dates > last_date]

        if future_dates.empty:
            continue

        if future_dates[0].to_period("Q") == quarter:
            continue

        window_start = last_date - pd.DateOffset(
            months=lookback_months
        )

        # Require a complete historical window.
        if dates[0] > window_start:
            continue

        training_dates = dates[
            (dates > window_start)
            & (dates <= last_date)
        ]

        if len(training_dates) < 2:
            continue

        rebalance_dates.append(last_date)

    return pd.DatetimeIndex(rebalance_dates)



def simulate_portfolio(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    initial_capital: float = 100_000.0,
    transaction_cost_bps: float = 0.0,
) -> dict:
    """
    Simulate a portfolio with drifting weights.

    Parameters
    ----------
    returns:
        Daily simple asset returns.

    target_weights:
        Target allocations indexed by rebalance decision dates.

    initial_capital:
        Starting portfolio value.

    transaction_cost_bps:
        Proportional transaction cost in basis points
        of traded notional.

    Assumptions
    -----------
    Rebalances execute at the decision-date close.

    The decision-date return is earned using the
    previous holdings. Rebalancing happens afterward.

    Newly allocated capital earns returns starting
    on the following trading session.

    Transaction costs are deducted at each rebalance.

    Returns
    -------
    dict containing:
        nav
        portfolio_returns
        weights
        turnover
        costs
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a DataFrame")

    if not isinstance(target_weights, pd.DataFrame):
        raise TypeError("target_weights must be a DataFrame")

    if returns.empty or target_weights.empty:
        raise ValueError("inputs cannot be empty")

    if (
        not returns.index.is_unique
        or not returns.index.is_monotonic_increasing
        or not target_weights.index.is_unique
        or not target_weights.index.is_monotonic_increasing
    ):
        raise ValueError("dates must be sorted and unique")

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns index must be a DatetimeIndex")

    if not isinstance(target_weights.index, pd.DatetimeIndex):
        raise TypeError(
            "target_weights index must be a DatetimeIndex"
        )

    if returns.index.hasnans or target_weights.index.hasnans:
        raise ValueError("dates cannot contain NaT")

    if not returns.columns.is_unique:
        raise ValueError("asset labels must be unique")

    if (
        not target_weights.columns.is_unique
        or set(returns.columns) != set(target_weights.columns)
    ):
        raise ValueError("asset labels do not match")

    if not target_weights.index.isin(returns.index).all():
        raise ValueError(
            "rebalance dates must exist in returns"
        )

    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("initial_capital must be positive")

    if (
        not np.isfinite(transaction_cost_bps)
        or transaction_cost_bps < 0
    ):
        raise ValueError(
            "transaction_cost_bps must be non-negative"
        )

    returns = returns.astype(float)

    target_weights = target_weights.reindex(
        columns=returns.columns
    ).astype(float)

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("returns must be finite")

    if not np.isfinite(target_weights.to_numpy()).all():
        raise ValueError("weights must be finite")

    if (returns < -1).any().any():
        raise ValueError("simple returns cannot be below -100%")

    if (target_weights < -1e-10).any().any():
        raise ValueError("negative weights are not allowed")

    if not np.allclose(
        target_weights.sum(axis=1),
        1.0,
        atol=1e-8,
    ):
        raise ValueError("weights must sum to one")

    first_rebalance = target_weights.index[0]

    # Start at the first decision date.
    simulation_dates = returns.index[
        returns.index >= first_rebalance
    ]

    assets = returns.columns

    holdings = pd.Series(
        0.0,
        index=assets,
        dtype=float,
    )

    cash = float(initial_capital)

    nav_history = {}
    weights_history = {}
    turnover_history = {}
    costs_history = {}

    cost_rate = transaction_cost_bps / 10_000

    for date in simulation_dates:

        daily_returns = returns.loc[date]

        # 1. Existing holdings earn today's returns.
        # The first decision date starts entirely in cash.
        holdings = holdings * (1.0 + daily_returns)

        portfolio_value = float(holdings.sum() + cash)

        turnover = 0.0
        cost = 0.0

        # 2. Rebalance after today's market return.
        if date in target_weights.index:

            target = target_weights.loc[date]

            (
                holdings,
                cash,
                turnover,
                cost,
            ) = _rebalance_with_costs(
                holdings=holdings,
                cash=cash,
                target=target,
                cost_rate=cost_rate,
            )

            portfolio_value = float(
                holdings.sum() + cash
            )

        # 3. Record end-of-day portfolio value.
        nav_history[date] = portfolio_value

        weights_history[date] = (
            holdings / portfolio_value
        )

        turnover_history[date] = turnover
        costs_history[date] = cost

    nav = pd.Series(
        nav_history,
        name="nav",
        dtype=float,
    )

    portfolio_returns = nav.pct_change()

    # Include initial investment costs in the first return.
    portfolio_returns.iloc[0] = (
        nav.iloc[0] / initial_capital - 1.0
    )

    portfolio_returns.name = "portfolio_return"

    weights = pd.DataFrame.from_dict(
        weights_history,
        orient="index",
    ).reindex(columns=assets)

    turnover = pd.Series(
        turnover_history,
        name="turnover",
    )

    costs = pd.Series(
        costs_history,
        name="transaction_cost",
    )

    return {
        "nav": nav,
        "portfolio_returns": portfolio_returns,
        "weights": weights,
        "turnover": turnover,
        "costs": costs,
    }



def _rebalance_with_costs(
    holdings: pd.Series,
    cash: float,
    target: pd.Series,
    cost_rate: float,
) -> tuple[pd.Series, float, float, float]:
    """
    Rebalance holdings with exact proportional trading costs.

    Returns
    -------
    new_holdings
    new_cash
    turnover
    transaction_cost
    """
    pre_nav = float(holdings.sum() + cash)

    if pre_nav <= 0:
        raise ValueError("portfolio value must be positive")

    if cost_rate == 0:
        post_nav = pre_nav
    else:
        def equation(post_nav):
            desired = target * post_nav
            traded = (desired - holdings).abs().sum()
            return post_nav + cost_rate * traded - pre_nav

        if equation(0.0) > 0:
            raise ValueError("transaction costs exhaust portfolio")

        post_nav = brentq(
            equation,
            0.0,
            pre_nav,
            xtol=1e-10,
        )

    new_holdings = target * post_nav

    traded_notional = float(
        (new_holdings - holdings).abs().sum()
    )

    transaction_cost = (
        cost_rate * traded_notional
    )

    turnover = traded_notional / pre_nav

    return (
        new_holdings,
        0.0,
        turnover,
        transaction_cost,
    )