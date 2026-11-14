import numpy as np
import pandas as pd
import pytest

from quantmgmt.backtesting.walk_forward import (
    generate_walk_forward_with_fallback,
)

from quantmgmt.optimizers.mean_variance import (
    OptimizationError,
)

from quantmgmt.backtesting.engine import (
    simulate_portfolio,
)


@pytest.fixture
def historical_returns():
    dates = pd.bdate_range(
        "2012-01-02",
        "2018-12-31",
    )

    return pd.DataFrame(
        {
            "A": np.full(len(dates), 0.001),
            "B": np.full(len(dates), -0.0005),
        },
        index=dates,
    )


def test_initial_failure_uses_equal_weights(historical_returns):

    def failing_strategy(window, date):
        raise OptimizationError("solver did not converge")

    weights, incidents = generate_walk_forward_with_fallback(
        historical_returns,
        failing_strategy,
    )

    assert len(weights) == 1

    np.testing.assert_allclose(
        weights.iloc[0].to_numpy(),
        [0.5, 0.5],
    )

    assert incidents.iloc[0]["action"] == "equal_weight_initial"


def test_later_failure_skips_rebalance(historical_returns):

    calls = 0

    def strategy(window, date):
        nonlocal calls
        calls += 1

        if calls == 2:
            raise OptimizationError("solver failed")

        return pd.Series({
            "A": 0.6,
            "B": 0.4,
        })

    weights, incidents = generate_walk_forward_with_fallback(
        historical_returns,
        strategy,
    )

    failed_date = incidents.iloc[0]["date"]

    assert failed_date not in weights.index
    assert incidents.iloc[0]["action"] == "hold_positions"

    result = simulate_portfolio(
        historical_returns,
        weights,
    )

    assert result["turnover"].loc[failed_date] == pytest.approx(0.0)


def test_programming_error_is_not_hidden(historical_returns):

    def broken_strategy(window, date):
        raise RuntimeError("unexpected bug")

    with pytest.raises(RuntimeError, match="unexpected bug"):
        generate_walk_forward_with_fallback(
            historical_returns,
            broken_strategy,
        )


def test_invalid_weights_are_not_recovered(historical_returns):

    def invalid_strategy(window, date):
        return pd.Series({
            "A": 0.9,
            "B": 0.9,
        })

    with pytest.raises(ValueError, match="sum to one"):
        generate_walk_forward_with_fallback(
            historical_returns,
            invalid_strategy,
        )