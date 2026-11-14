import numpy as np
import pandas as pd
import pytest

from quantmgmt.backtesting.views import (
    three_relative_views,
)
from quantmgmt.backtesting.comparison import (
    run_strategy_comparison,
)


@pytest.fixture
def historical_returns():

    dates = pd.bdate_range(
        "2020-01-01",
        "2024-12-31",
    )

    rng = np.random.default_rng(42)

    assets = [
        "SPY", "IWM", "EFA", "EEM",
        "TLT", "IEF", "LQD", "HYG",
        "GLD", "DBC", "VNQ", "TIP",
    ]

    return pd.DataFrame(
        rng.normal(
            0.0003,
            0.01,
            size=(len(dates), len(assets)),
        ),
        index=dates,
        columns=assets,
    )


def test_three_views_shapes(historical_returns):

    P, Q, confidences = three_relative_views(
        historical_returns,
        historical_returns.index[-1],
    )

    assert P.shape == (3, 12)
    assert Q.shape == (3,)
    assert confidences.shape == (3,)

    assert P.index.equals(Q.index)
    assert P.index.equals(confidences.index)

    assert P.columns.equals(historical_returns.columns)


def test_three_views_strategic_exposures(historical_returns):

    P, Q, confidences = three_relative_views(
        historical_returns,
        historical_returns.index[-1],
    )

    gld_view = P.loc["GLD_outperforms_DBC"]

    assert gld_view["GLD"] == pytest.approx(1.0)
    assert gld_view["DBC"] == pytest.approx(-1.0)

    assert gld_view.abs().sum() == pytest.approx(2.0)

    spy_view = P.loc["SPY_outperforms_EFA"]

    assert spy_view["SPY"] == pytest.approx(1.0)
    assert spy_view["EFA"] == pytest.approx(-1.0)

    assert spy_view.abs().sum() == pytest.approx(2.0)

    assert Q.loc["GLD_outperforms_DBC"] == pytest.approx(0.01)
    assert Q.loc["SPY_outperforms_EFA"] == pytest.approx(0.015)

    assert confidences.loc[
        "GLD_outperforms_DBC"
    ] == pytest.approx(0.55)


def test_three_views_momentum_no_lookahead(historical_returns):

    date = pd.Timestamp("2023-12-29")

    # Only information available at the decision date.
    past_returns = historical_returns.loc[
        historical_returns.index <= date
    ].copy()

    P1, Q1, C1 = three_relative_views(
        past_returns,
        date,
    )

    # Modify future observations in the complete dataset.
    modified = historical_returns.copy()

    modified.loc[
        modified.index > date, "SPY"
    ] = 0.50

    # The walk-forward engine must supply only past data.
    modified_past = modified.loc[
        modified.index <= date
    ].copy()

    P2, Q2, C2 = three_relative_views(
        modified_past,
        date,
    )

    # Decisions must remain identical.
    pd.testing.assert_frame_equal(P1, P2)
    pd.testing.assert_series_equal(Q1, Q2)
    pd.testing.assert_series_equal(C1, C2)


def test_three_views_requires_strategic_assets(
    historical_returns,
):

    incomplete = historical_returns.drop(
        columns=["GLD"]
    )

    with pytest.raises(ValueError, match="missing assets"):
        three_relative_views(
            incomplete,
            incomplete.index[-1],
        )



def test_three_views_rejects_future_data(historical_returns):

    date = pd.Timestamp("2023-12-29")

    with pytest.raises(
        ValueError,
        match="future information detected",
    ):
        three_relative_views(
            historical_returns,
            date,
        )






def test_comparison_with_three_views(historical_returns):

    benchmark = pd.Series(
        1.0 / len(historical_returns.columns),
        index=historical_returns.columns,
    )

    result = run_strategy_comparison(
        returns=historical_returns,
        market_weights=benchmark,
        bl_view_mode="three",
    )

    assert "black_litterman" in result["results"]

    weights = result["weights"]["black_litterman"]

    assert not weights.empty

    np.testing.assert_allclose(
        weights.sum(axis=1).to_numpy(),
        np.ones(len(weights)),
        atol=1e-8,
    )



@pytest.mark.parametrize(
    "include_gld,include_spy,expected_count",
    [
        (False, False, 1),
        (True, False, 2),
        (False, True, 2),
        (True, True, 3),
    ],
)
def test_selectable_views(
    historical_returns,
    include_gld,
    include_spy,
    expected_count,
):

    P, Q, C = three_relative_views(
        historical_returns,
        historical_returns.index[-1],
        include_gld_view=include_gld,
        include_spy_view=include_spy,
    )

    assert len(P) == expected_count
    assert len(Q) == expected_count
    assert len(C) == expected_count

    assert P.index.equals(Q.index)
    assert P.index.equals(C.index)