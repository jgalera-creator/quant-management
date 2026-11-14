import numpy as np
import pandas as pd
import pytest

from quantmgmt.backtesting.views import (
    momentum_relative_view,
)


@pytest.fixture
def sample_returns():
    dates = pd.bdate_range(
        "2020-01-01",
        "2021-12-31",
    )

    return pd.DataFrame(
        {
            "A": np.full(len(dates), 0.001),
            "B": np.full(len(dates), -0.001),
            "C": np.zeros(len(dates)),
        },
        index=dates,
    )


def test_momentum_selects_winner_and_loser(sample_returns):
    date = sample_returns.index[-1]

    P, Q, confidences = momentum_relative_view(
        sample_returns,
        date,
    )

    assert P.iloc[0]["A"] == pytest.approx(1.0)
    assert P.iloc[0]["B"] == pytest.approx(-1.0)
    assert P.iloc[0]["C"] == pytest.approx(0.0)

    assert Q.iloc[0] == pytest.approx(0.03)
    assert confidences.iloc[0] == pytest.approx(0.60)


def test_momentum_view_is_relative(sample_returns):
    P, _, _ = momentum_relative_view(
        sample_returns,
        sample_returns.index[-1],
    )

    assert P.iloc[0].sum() == pytest.approx(0.0)


def test_momentum_does_not_use_future_returns(sample_returns):
    cutoff = pd.Timestamp("2021-06-30")

    historical = sample_returns.loc[:cutoff].copy()

    P1, Q1, C1 = momentum_relative_view(
        historical,
        cutoff,
    )

    modified = sample_returns.copy()

    modified.loc[
        modified.index > cutoff, "B"
    ] = 0.50

    P2, Q2, C2 = momentum_relative_view(
        modified.loc[:cutoff],
        cutoff,
    )

    pd.testing.assert_frame_equal(P1, P2)
    pd.testing.assert_series_equal(Q1, Q2)
    pd.testing.assert_series_equal(C1, C2)


def test_momentum_rejects_future_information(sample_returns):
    cutoff = pd.Timestamp("2021-06-30")

    with pytest.raises(ValueError, match="future"):
        momentum_relative_view(
            sample_returns,
            cutoff,
        )


def test_momentum_rejects_invalid_confidence(sample_returns):
    with pytest.raises(ValueError, match="confidence"):
        momentum_relative_view(
            sample_returns,
            sample_returns.index[-1],
            confidence=1.5,
        )


def test_momentum_requires_multiple_assets(sample_returns):
    with pytest.raises(ValueError, match="two assets"):
        momentum_relative_view(
            sample_returns[["A"]],
            sample_returns.index[-1],
        )



def three_relative_views(
    historical_returns: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    momentum_months: int = 12,
    expected_spread: float = 0.03,
    confidence: float = 0.60,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Construct three Black-Litterman relative views.

    V1: Dynamic 12-month momentum.
    V2: GLD outperforms DBC by 1% annually.
    V3: SPY outperforms EFA by 1.5% annually.

    All Q values are annualized excess-return spreads.
    """

    required_assets = {"GLD", "DBC", "SPY", "EFA"}

    if not isinstance(historical_returns, pd.DataFrame):
        raise TypeError(
            "historical_returns must be a DataFrame"
        )

    missing = required_assets - set(
        historical_returns.columns
    )

    if missing:
        raise ValueError(
            f"missing assets for strategic views: {sorted(missing)}"
        )

    # V1: Use the existing, validated momentum model.
    P_momentum, Q_momentum, C_momentum = (
        momentum_relative_view(
            historical_returns=historical_returns,
            rebalance_date=rebalance_date,
            momentum_months=momentum_months,
            expected_spread=expected_spread,
            confidence=confidence,
        )
    )

    assets = historical_returns.columns

    # Build strategic relative views.
    P_strategic = pd.DataFrame(
        0.0,
        index=[
            "GLD_outperforms_DBC",
            "SPY_outperforms_EFA",
        ],
        columns=assets,
    )

    P_strategic.loc[
        "GLD_outperforms_DBC", "GLD"
    ] = 1.0

    P_strategic.loc[
        "GLD_outperforms_DBC", "DBC"
    ] = -1.0

    P_strategic.loc[
        "SPY_outperforms_EFA", "SPY"
    ] = 1.0

    P_strategic.loc[
        "SPY_outperforms_EFA", "EFA"
    ] = -1.0

    Q_strategic = pd.Series(
        {
            "GLD_outperforms_DBC": 0.01,
            "SPY_outperforms_EFA": 0.015,
        },
        name="expected_spread",
    )

    C_strategic = pd.Series(
        {
            "GLD_outperforms_DBC": 0.55,
            "SPY_outperforms_EFA": 0.55,
        },
        name="confidence",
    )

    # Avoid collisions when the dynamic view coincides
    # with one of the strategic views.
    P_momentum = P_momentum.copy()
    Q_momentum = Q_momentum.copy()
    C_momentum = C_momentum.copy()

    P_momentum.index = ["momentum_relative"]
    Q_momentum.index = ["momentum_relative"]
    C_momentum.index = ["momentum_relative"]

    P = pd.concat(
        [P_momentum, P_strategic],
        axis=0,
    )

    Q = pd.concat(
        [Q_momentum, Q_strategic],
        axis=0,
    )

    confidences = pd.concat(
        [C_momentum, C_strategic],
        axis=0,
    )

    # Explicit alignment.
    Q = Q.reindex(P.index)
    confidences = confidences.reindex(P.index)

    if P.shape != (3, len(assets)):
        raise ValueError("invalid view matrix shape")

    if not P.index.is_unique:
        raise ValueError("duplicate view labels")

    if not (
        np.isfinite(P.to_numpy()).all()
        and np.isfinite(Q.to_numpy()).all()
        and np.isfinite(confidences.to_numpy()).all()
    ):
        raise ValueError("non-finite view parameters")

    return P, Q, confidences