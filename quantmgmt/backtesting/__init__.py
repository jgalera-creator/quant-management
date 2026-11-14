from collections.abc import Callable

import numpy as np
import pandas as pd

from quantmgmt.backtesting.engine import (
    quarterly_rebalance_dates,
)


WeightFunction = Callable[
    [pd.DataFrame, pd.Timestamp],
    pd.Series,
]


def generate_walk_forward_weights(
    returns: pd.DataFrame,
    weight_function: WeightFunction,
    lookback_months: int = 36,
) -> pd.DataFrame:
    """
    Generate target weights using walk-forward estimation.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily simple returns.

        Rows: trading dates.
        Columns: assets.

    weight_function : callable
        Function receiving:

            historical_returns, rebalance_date

        It must return a pd.Series indexed by asset.

    lookback_months : int
        Historical estimation window in calendar months.

    Returns
    -------
    pd.DataFrame
        Target weights indexed by rebalance date.

    Notes
    -----
    Only observations available up to each rebalance
    date are supplied to the weight function.

    The returned weights are decision-date weights.
    Execution is handled separately by the simulator.
    """

    if not isinstance(returns, pd.DataFrame):
        raise TypeError(
            "returns must be a pandas DataFrame"
        )

    if returns.empty:
        raise ValueError(
            "returns cannot be empty"
        )

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError(
            "returns index must be a DatetimeIndex"
        )

    if (
        not returns.index.is_unique
        or not returns.index.is_monotonic_increasing
    ):
        raise ValueError(
            "dates must be sorted and unique"
        )

    if returns.index.hasnans:
        raise ValueError(
            "dates cannot contain NaT"
        )

    if not returns.columns.is_unique:
        raise ValueError(
            "asset labels must be unique"
        )

    if not np.isfinite(
        returns.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "returns must be finite"
        )

    if not callable(weight_function):
        raise TypeError(
            "weight_function must be callable"
        )

    if lookback_months <= 0:
        raise ValueError(
            "lookback_months must be positive"
        )

    assets = returns.columns

    rebalance_dates = quarterly_rebalance_dates(
        returns.index,
        lookback_months=lookback_months,
    )

    weights_history = {}

    for date in rebalance_dates:

        window_start = (
            date
            - pd.DateOffset(months=lookback_months)
        )

        historical_returns = returns.loc[
            (returns.index > window_start)
            & (returns.index <= date)
        ].copy()

        # Explicit look-ahead protection.
        if historical_returns.index.max() > date:
            raise ValueError(
                "look-ahead bias detected"
            )

        # A separate copy prevents callbacks from
        # modifying the original return matrix.
        weights = weight_function(
            historical_returns.copy(),
            date,
        )

        if not isinstance(weights, pd.Series):
            raise TypeError(
                "weight_function must return a Series"
            )

        if not weights.index.is_unique:
            raise ValueError(
                "weight labels must be unique"
            )

        if set(weights.index) != set(assets):
            raise ValueError(
                "weight assets do not match returns"
            )

        weights = weights.reindex(assets)

        if not np.isfinite(
            weights.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                "weights must be finite"
            )

        if (weights < -1e-8).any():
            raise ValueError(
                "negative weights are not allowed"
            )

        if not np.isclose(
            weights.sum(),
            1.0,
            atol=1e-8,
        ):
            raise ValueError(
                "weights must sum to one"
            )

        weights_history[date] = weights.copy()

    return pd.DataFrame.from_dict(
        weights_history,
        orient="index",
        columns=assets,
    ).rename_axis("Date")