import numpy as np
import pandas as pd


def momentum_relative_view(
    historical_returns: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    momentum_months: int = 12,
    expected_spread: float = 0.03,
    confidence: float = 0.60,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Generate one relative Black-Litterman view from past momentum.

    Select the strongest and weakest assets using trailing
    cumulative returns, then express the view that the
    strongest asset will outperform the weakest.

    Returns
    -------
    P : pd.DataFrame
        View exposure matrix.
    Q : pd.Series
        Annualized expected excess-return spread.
    confidences : pd.Series
        Confidence parameter for the view.
    """
    if not isinstance(historical_returns, pd.DataFrame):
        raise TypeError("historical_returns must be a DataFrame")

    if historical_returns.empty:
        raise ValueError("historical_returns cannot be empty")

    if not isinstance(historical_returns.index, pd.DatetimeIndex):
        raise TypeError("returns index must be a DatetimeIndex")

    if not historical_returns.index.is_monotonic_increasing:
        raise ValueError("dates must be sorted")

    if not historical_returns.index.is_unique:
        raise ValueError("dates must be unique")

    if not historical_returns.columns.is_unique:
        raise ValueError("asset labels must be unique")

    if historical_returns.shape[1] < 2:
        raise ValueError("at least two assets are required")

    if not np.isfinite(
        historical_returns.to_numpy(dtype=float)
    ).all():
        raise ValueError("returns must be finite")

    if momentum_months <= 0:
        raise ValueError("momentum_months must be positive")

    if not np.isfinite(expected_spread) or expected_spread <= 0:
        raise ValueError("expected_spread must be positive")

    if not np.isfinite(confidence) or not 0 < confidence <= 1:
        raise ValueError("confidence must be in (0, 1]")

    rebalance_date = pd.Timestamp(rebalance_date)

    if historical_returns.index.max() > rebalance_date:
        raise ValueError("future information detected")

    start_date = rebalance_date - pd.DateOffset(
        months=momentum_months
    )

    window = historical_returns.loc[
        (historical_returns.index > start_date)
        & (historical_returns.index <= rebalance_date)
    ]

    if len(window) < 2:
        raise ValueError("insufficient momentum history")

    momentum = (1.0 + window).prod() - 1.0

    winner = momentum.idxmax()
    loser = momentum.idxmin()

    if winner == loser:
        raise ValueError("cannot construct relative view")

    view_name = f"{winner}_outperforms_{loser}"

    P = pd.DataFrame(
        0.0,
        index=[view_name],
        columns=historical_returns.columns,
    )

    P.loc[view_name, winner] = 1.0
    P.loc[view_name, loser] = -1.0

    Q = pd.Series(
        {view_name: expected_spread},
        name="expected_spread",
    )

    confidences = pd.Series(
        {view_name: confidence},
        name="confidence",
    )

    return P, Q, confidences