import pandas as pd
import pytest
import numpy as np

from quantmgmt.backtesting.engine import (
    quarterly_rebalance_dates,
    simulate_portfolio,
)


def test_quarterly_rebalance_dates():
    dates = pd.bdate_range(
        "2020-01-01",
        "2024-12-31",
    )

    result = quarterly_rebalance_dates(
        dates,
        lookback_months=36,
    )

    assert len(result) > 0
    assert result.is_monotonic_increasing
    assert result.is_unique

    # Solo un rebalanceo por trimestre
    assert result.to_period("Q").is_unique

    # La primera fecha debe tener suficiente historial
    assert result[0] >= dates[0] + pd.DateOffset(months=36)

    # El último trimestre incompleto no debe incluirse
    assert result[-1] < dates[-1]


def test_rebalance_requires_sufficient_history():
    dates = pd.bdate_range(
        "2020-01-01",
        periods=100,
    )

    result = quarterly_rebalance_dates(
        dates,
        lookback_months=36,
    )

    assert len(result) == 0



def test_rebalance_dates_have_future_session():
    dates = pd.bdate_range(
        "2020-01-01",
        "2023-12-31",
    )

    result = quarterly_rebalance_dates(
        dates,
        lookback_months=36,
    )

    assert len(result) > 0

    # Siempre debe existir una sesión posterior
    assert all(date < dates[-1] for date in result)

    # La sesión siguiente debe pertenecer
    # a un trimestre distinto.
    for date in result:
        position = dates.get_loc(date)
        next_date = dates[position + 1]

        assert next_date.to_period("Q") != date.to_period("Q")


def test_rebalance_rejects_unsorted_dates():
    dates = pd.DatetimeIndex([
        "2023-01-03",
        "2023-01-02",
    ])

    with pytest.raises(ValueError, match="sorted"):
        quarterly_rebalance_dates(dates)


def test_rebalance_rejects_duplicates():
    dates = pd.DatetimeIndex([
        "2023-01-02",
        "2023-01-02",
        "2023-01-03",
    ])

    with pytest.raises(ValueError, match="unique"):
        quarterly_rebalance_dates(dates)



def test_portfolio_weights_drift():

    dates = pd.bdate_range(
        "2024-01-01",
        periods=3,
    )

    returns = pd.DataFrame(
        {
            "A": [0.0, 0.20, 0.0],
            "B": [0.0, -0.10, 0.0],
        },
        index=dates,
    )

    targets = pd.DataFrame(
        {
            "A": [0.5],
            "B": [0.5],
        },
        index=[dates[0]],
    )

    result = simulate_portfolio(
        returns,
        targets,
        initial_capital=10_000,
    )

    assert result["nav"].iloc[1] == pytest.approx(
        10_500
    )

    weights = result["weights"].iloc[1]

    assert weights["A"] == pytest.approx(
        6000 / 10500
    )

    assert weights["B"] == pytest.approx(
        4500 / 10500
    )



def test_rebalance_restores_target_weights():

    dates = pd.bdate_range(
        "2024-01-01",
        periods=4,
    )

    returns = pd.DataFrame(
        {
            "A": [0.0, 0.20, 0.0, 0.0],
            "B": [0.0, -0.10, 0.0, 0.0],
        },
        index=dates,
    )

    targets = pd.DataFrame(
        {
            "A": [0.5, 0.5],
            "B": [0.5, 0.5],
        },
        index=[dates[0], dates[2]],
    )

    result = simulate_portfolio(
        returns,
        targets,
    )

    weights = result["weights"].loc[dates[2]]

    assert weights["A"] == pytest.approx(0.5)
    assert weights["B"] == pytest.approx(0.5)

    assert result["turnover"].loc[dates[2]] > 0



def test_transaction_costs_reduce_nav():

    dates = pd.bdate_range(
        "2024-01-01",
        periods=3,
    )

    returns = pd.DataFrame(
        {
            "A": [0.0, 0.01, 0.01],
            "B": [0.0, 0.01, 0.01],
        },
        index=dates,
    )

    targets = pd.DataFrame(
        {
            "A": [0.5],
            "B": [0.5],
        },
        index=[dates[0]],
    )

    without_costs = simulate_portfolio(
        returns,
        targets,
        transaction_cost_bps=0.0,
    )

    with_costs = simulate_portfolio(
        returns,
        targets,
        transaction_cost_bps=10.0,
    )

    assert (
        with_costs["nav"].iloc[-1]
        < without_costs["nav"].iloc[-1]
    )



def test_first_rebalance_does_not_capture_same_day_return():

    dates = pd.bdate_range(
        "2024-01-01",
        periods=3,
    )

    returns = pd.DataFrame(
        {
            "A": [0.50, 0.10, 0.0],
        },
        index=dates,
    )

    targets = pd.DataFrame(
        {
            "A": [1.0],
        },
        index=[dates[0]],
    )

    result = simulate_portfolio(
        returns,
        targets,
        initial_capital=1000,
    )

    assert result["nav"].iloc[0] == pytest.approx(1000)

    assert result["nav"].iloc[1] == pytest.approx(1100)


def test_simulated_weights_sum_to_one():

    dates = pd.bdate_range(
        "2024-01-01",
        periods=10,
    )

    returns = pd.DataFrame(
        {
            "A": np.linspace(0.001, 0.01, 10),
            "B": np.linspace(-0.002, 0.005, 10),
            "C": np.zeros(10),
        },
        index=dates,
    )

    targets = pd.DataFrame(
        {
            "A": [0.5],
            "B": [0.3],
            "C": [0.2],
        },
        index=[dates[0]],
    )

    result = simulate_portfolio(
        returns,
        targets,
    )

    np.testing.assert_allclose(
        result["weights"].sum(axis=1),
        1.0,
        atol=1e-10,
    )



def test_initial_investment_exact_cost():

    dates = pd.bdate_range(
        "2024-01-01",
        periods=2,
    )

    returns = pd.DataFrame(
        {"A": [0.0, 0.0]},
        index=dates,
    )

    targets = pd.DataFrame(
        {"A": [1.0]},
        index=[dates[0]],
    )

    result = simulate_portfolio(
        returns,
        targets,
        initial_capital=1000,
        transaction_cost_bps=100,
    )

    # Cost rate = 1%.
    # Exact post-trade NAV = 1000 / 1.01.
    expected_nav = 1000 / 1.01

    assert result["nav"].iloc[0] == pytest.approx(
        expected_nav
    )



def test_rebalance_accounting_identity():

    dates = pd.bdate_range(
        "2024-01-01",
        periods=3,
    )

    returns = pd.DataFrame(
        {
            "A": [0.0, 0.20, 0.0],
            "B": [0.0, -0.10, 0.0],
        },
        index=dates,
    )

    targets = pd.DataFrame(
        {
            "A": [0.5, 0.5],
            "B": [0.5, 0.5],
        },
        index=[dates[0], dates[2]],
    )

    result = simulate_portfolio(
        returns,
        targets,
        initial_capital=10_000,
        transaction_cost_bps=10,
    )

    # After the second rebalance, NAV plus costs
    # must equal pre-trade portfolio value.
    nav_before = result["nav"].iloc[1]

    nav_after = result["nav"].iloc[2]

    costs = result["costs"].iloc[2]

    assert nav_after + costs == pytest.approx(
        nav_before,
        abs=1e-6,
    )



def test_no_trade_when_weights_unchanged():

    dates = pd.bdate_range(
        "2024-01-01",
        periods=3,
    )

    returns = pd.DataFrame(
        {
            "A": [0.0, 0.0, 0.0],
            "B": [0.0, 0.0, 0.0],
        },
        index=dates,
    )

    targets = pd.DataFrame(
        {
            "A": [0.5, 0.5],
            "B": [0.5, 0.5],
        },
        index=[dates[0], dates[2]],
    )

    result = simulate_portfolio(
        returns,
        targets,
        transaction_cost_bps=10,
    )

    assert result["turnover"].iloc[2] == pytest.approx(
        0.0,
        abs=1e-10,
    )

    assert result["costs"].iloc[2] == pytest.approx(
        0.0,
        abs=1e-10,
    )



def test_initial_transaction_cost_in_returns():

    dates = pd.bdate_range(
        "2024-01-01",
        periods=3,
    )

    returns = pd.DataFrame(
        {"A": [0.0, 0.0, 0.0]},
        index=dates,
    )

    targets = pd.DataFrame(
        {"A": [1.0]},
        index=[dates[0]],
    )

    result = simulate_portfolio(
        returns=returns,
        target_weights=targets,
        initial_capital=100_000,
        transaction_cost_bps=10,
    )

    expected_nav = 100_000 / 1.001

    expected_return = (
        expected_nav / 100_000 - 1
    )

    assert result["portfolio_returns"].iloc[0] == pytest.approx(
        expected_return
    )



def test_compounded_returns_match_final_nav():

    dates = pd.bdate_range(
        "2024-01-01",
        periods=5,
    )

    returns = pd.DataFrame(
        {
            "A": [0.0, 0.02, -0.01, 0.03, 0.01],
            "B": [0.0, -0.01, 0.02, 0.01, -0.02],
        },
        index=dates,
    )

    targets = pd.DataFrame(
        {
            "A": [0.6, 0.4],
            "B": [0.4, 0.6],
        },
        index=[dates[0], dates[3]],
    )

    result = simulate_portfolio(
        returns=returns,
        target_weights=targets,
        initial_capital=100_000,
        transaction_cost_bps=10,
    )

    compounded = (
        1.0 + result["portfolio_returns"]
    ).prod()

    final_nav_ratio = (
        result["nav"].iloc[-1] / 100_000
    )

    assert compounded == pytest.approx(
        final_nav_ratio,
        rel=1e-10,
    )



def test_zero_returns_zero_costs():

    dates = pd.bdate_range(
        "2024-01-01",
        periods=5,
    )

    returns = pd.DataFrame(
        0.0,
        index=dates,
        columns=["A", "B"],
    )

    targets = pd.DataFrame(
        {
            "A": [0.5],
            "B": [0.5],
        },
        index=[dates[0]],
    )

    result = simulate_portfolio(
        returns=returns,
        target_weights=targets,
        initial_capital=100_000,
        transaction_cost_bps=0,
    )

    np.testing.assert_allclose(
        result["portfolio_returns"],
        0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        result["nav"],
        100_000,
        atol=1e-8,
    )


def test_rebalance_rejects_incomplete_quarter():

    dates = pd.bdate_range(
        "2019-01-01",
        "2024-06-30",
    )

    # Simulate missing observations near
    # the end of March 2023.
    dates = dates[
        ~(
            (dates >= "2023-03-20")
            & (dates <= "2023-03-31")
        )
    ]

    result = quarterly_rebalance_dates(
        dates,
        lookback_months=36,
    )

    assert pd.Timestamp("2023-03-17") not in result

    assert not any(
        date.to_period("Q") == pd.Period("2023Q1")
        for date in result
    )



def test_rebalance_accepts_valid_quarter_end():

    dates = pd.bdate_range(
        "2019-01-01",
        "2024-06-30",
    )

    result = quarterly_rebalance_dates(
        dates,
        lookback_months=36,
    )

    assert pd.Timestamp("2023-03-31") in result



def test_rebalance_has_full_lookback():

    dates = pd.bdate_range(
        "2012-01-02",
        "2020-12-31",
    )

    result = quarterly_rebalance_dates(
        dates,
        lookback_months=36,
    )

    assert len(result) > 0

    for date in result:

        start = date - pd.DateOffset(months=36)

        assert dates[0] <= start

        training = dates[
            (dates > start)
            & (dates <= date)
        ]

        assert len(training) > 0
        assert training.max() <= date



