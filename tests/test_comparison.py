import numpy as np
import pandas as pd
import pytest

from quantmgmt.backtesting.comparison import (
    STRATEGIES,
    run_strategy_comparison,
    simulate_portfolio,
)


@pytest.fixture(scope="module")
def comparison_result():

    rng = np.random.default_rng(42)

    dates = pd.bdate_range(
        "2012-01-02",
        "2017-12-29",
    )

    assets = ["A", "B", "C"]

    returns = pd.DataFrame(
        rng.normal(
            loc=[0.0005, 0.0003, 0.0002],
            scale=[0.01, 0.008, 0.006],
            size=(len(dates), 3),
        ),
        index=dates,
        columns=assets,
    )

    benchmark = pd.Series({
        "A": 0.5,
        "B": 0.3,
        "C": 0.2,
    })

    return run_strategy_comparison(
        returns=returns,
        market_weights=benchmark,
        risk_aversion=2.5,
        lookback_months=36,
    )


def test_comparison_contains_five_strategies(
    comparison_result,
):

    assert set(comparison_result["weights"]) == set(
        STRATEGIES
    )


def test_comparison_nav_columns(
    comparison_result,
):

    nav = comparison_result["nav"]

    assert nav.columns.tolist() == list(STRATEGIES)


def test_comparison_nav_is_positive(
    comparison_result,
):

    nav = comparison_result["nav"]

    assert (nav > 0).all().all()


def test_comparison_weights_sum_to_one(
    comparison_result,
):

    for weights in comparison_result["weights"].values():

        np.testing.assert_allclose(
            weights.sum(axis=1),
            1.0,
            atol=1e-6,
        )


def test_comparison_common_dates(
    comparison_result,
):

    nav = comparison_result["nav"]

    for result in comparison_result["results"].values():

        assert result["nav"].index.equals(
            nav.index
        )


def test_comparison_no_missing_nav(
    comparison_result,
):

    assert not comparison_result["nav"].isna().any().any()


def test_comparison_equal_weight(
    comparison_result,
):

    weights = comparison_result["weights"]["equal_weight"]

    np.testing.assert_allclose(
        weights.to_numpy(),
        np.full(weights.shape, 1.0 / 3.0),
        atol=1e-10,
    )


def test_comparison_turnover_nonnegative(
    comparison_result,
):

    turnover = comparison_result["turnover"]

    assert (turnover >= 0).all().all()



def test_comparison_no_lookahead():
    rng = np.random.default_rng(123)

    dates = pd.bdate_range(
        "2012-01-02",
        "2017-12-29",
    )

    assets = ["A", "B", "C"]

    returns = pd.DataFrame(
        rng.normal(
            loc=[0.0005, 0.0003, 0.0002],
            scale=[0.01, 0.008, 0.006],
            size=(len(dates), 3),
        ),
        index=dates,
        columns=assets,
    )

    benchmark = pd.Series({
        "A": 0.5,
        "B": 0.3,
        "C": 0.2,
    })

    original = run_strategy_comparison(
        returns,
        benchmark,
    )

    modified_returns = returns.copy()

    # Alter future data without changing past observations.
    cutoff = pd.Timestamp("2017-01-01")

    modified_returns.loc[
        modified_returns.index >= cutoff, "A"
    ] = 0.02

    modified = run_strategy_comparison(
        modified_returns,
        benchmark,
    )

    for strategy in STRATEGIES:
        past_dates = original["weights"][strategy].index[
            original["weights"][strategy].index < cutoff
        ]

        pd.testing.assert_frame_equal(
            original["weights"][strategy].loc[past_dates],
            modified["weights"][strategy].loc[past_dates],
        )



def test_comparison_nav_matches_compounded_returns(
    comparison_result,
):
    initial_capital = 100_000.0

    for strategy in STRATEGIES:
        nav = comparison_result["nav"][strategy]
        returns = comparison_result["portfolio_returns"][strategy]

        compounded = (1.0 + returns).prod()

        final_nav_ratio = nav.iloc[-1] / initial_capital

        assert compounded == pytest.approx(
            final_nav_ratio,
            rel=1e-10,
        )



def test_transaction_costs_do_not_improve_fixed_strategy():
    rng = np.random.default_rng(42)

    dates = pd.bdate_range(
        "2020-01-01",
        periods=250,
    )

    returns = pd.DataFrame(
        rng.normal(
            0.0002,
            0.01,
            size=(250, 2),
        ),
        index=dates,
        columns=["A", "B"],
    )

    targets = pd.DataFrame(
        {
            "A": [0.5, 0.7],
            "B": [0.5, 0.3],
        },
        index=[dates[0], dates[125]],
    )

    without_costs = simulate_portfolio(
        returns,
        targets,
        transaction_cost_bps=0,
    )

    with_costs = simulate_portfolio(
        returns,
        targets,
        transaction_cost_bps=10,
    )

    assert (
        with_costs["nav"].iloc[-1]
        <= without_costs["nav"].iloc[-1]
    )



def test_all_strategies_start_same_day(
    comparison_result,
):
    starts = [
        result["nav"].index[0]
        for result in comparison_result["results"].values()
    ]

    assert len(set(starts)) == 1