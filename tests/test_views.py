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