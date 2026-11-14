import numpy as np
import pandas as pd
import pytest

from quantmgmt.backtesting.walk_forward import (
    generate_walk_forward_weights,
)


@pytest.fixture
def historical_returns():

    rng = np.random.default_rng(42)

    dates = pd.bdate_range(
        "2012-01-02",
        "2020-12-31",
    )

    data = rng.normal(
        loc=0.0003,
        scale=0.01,
        size=(len(dates), 3),
    )

    return pd.DataFrame(
        data,
        index=dates,
        columns=["A", "B", "C"],
    )


def test_walk_forward_weights_sum_to_one(
    historical_returns,
):

    def equal_weight(window, date):
        return pd.Series(
            1.0 / window.shape[1],
            index=window.columns,
        )

    weights = generate_walk_forward_weights(
        historical_returns,
        equal_weight,
        lookback_months=36,
    )

    assert len(weights) > 0

    np.testing.assert_allclose(
        weights.sum(axis=1),
        1.0,
        atol=1e-10,
    )



def test_walk_forward_no_future_data(
    historical_returns,
):

    observed_windows = []

    def strategy(window, date):

        observed_windows.append(
            (window.index.min(), window.index.max(), date)
        )

        assert window.index.max() <= date

        return pd.Series(
            1.0 / window.shape[1],
            index=window.columns,
        )

    generate_walk_forward_weights(
        historical_returns,
        strategy,
        lookback_months=36,
    )

    assert len(observed_windows) > 0

    for start, end, decision_date in observed_windows:

        assert end <= decision_date

        assert start > (
            decision_date
            - pd.DateOffset(months=36)
        )


def test_future_returns_do_not_change_past_weights(
    historical_returns,
):

    def momentum_strategy(window, date):

        cumulative = (
            (1.0 + window).prod() - 1.0
        )

        winner = cumulative.idxmax()

        weights = pd.Series(
            0.0,
            index=window.columns,
        )

        weights.loc[winner] = 1.0

        return weights

    original = generate_walk_forward_weights(
        historical_returns,
        momentum_strategy,
    )

    modified_returns = historical_returns.copy()

    # Deliberately alter future observations.
    modified_returns.loc[
        modified_returns.index >= "2020-01-01",
        "A",
    ] = 0.05

    modified = generate_walk_forward_weights(
        modified_returns,
        momentum_strategy,
    )

    past_dates = original.index[
        original.index < "2020-01-01"
    ]

    pd.testing.assert_frame_equal(
        original.loc[past_dates],
        modified.loc[past_dates],
    )



def test_walk_forward_preserves_asset_order(
    historical_returns,
):

    def reversed_weights(window, date):

        return pd.Series(
            {
                "C": 0.2,
                "B": 0.3,
                "A": 0.5,
            }
        )

    weights = generate_walk_forward_weights(
        historical_returns,
        reversed_weights,
    )

    assert weights.columns.tolist() == [
        "A", "B", "C"
    ]

    assert weights.iloc[0]["A"] == pytest.approx(0.5)



def test_walk_forward_rejects_invalid_weights(
    historical_returns,
):

    def invalid_strategy(window, date):

        return pd.Series(
            {
                "A": 0.6,
                "B": 0.6,
                "C": 0.2,
            }
        )

    with pytest.raises(
        ValueError,
        match="sum to one",
    ):
        generate_walk_forward_weights(
            historical_returns,
            invalid_strategy,
        )