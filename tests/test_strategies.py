import numpy as np
import pandas as pd
import pytest

from quantmgmt.backtesting.strategies import (
    compute_strategy_weights,
)


@pytest.fixture
def sample_returns():

    rng = np.random.default_rng(42)

    dates = pd.bdate_range(
        "2020-01-01",
        periods=500,
    )

    data = rng.normal(
        loc=[0.0004, 0.0003, 0.0002],
        scale=[0.012, 0.009, 0.006],
        size=(500, 3),
    )

    return pd.DataFrame(
        data,
        index=dates,
        columns=["A", "B", "C"],
    )


def test_equal_weight(sample_returns):

    weights = compute_strategy_weights(
        sample_returns,
        strategy="equal_weight",
    )

    np.testing.assert_allclose(
        weights.to_numpy(),
        np.full(3, 1 / 3),
    )


def test_min_variance(sample_returns):

    weights = compute_strategy_weights(
        sample_returns,
        strategy="min_variance",
    )

    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= -1e-8).all()


def test_max_sharpe(sample_returns):

    weights = compute_strategy_weights(
        sample_returns,
        strategy="max_sharpe",
        risk_free_rate=0.0,
    )

    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= -1e-8).all()


def test_hrp(sample_returns):

    weights = compute_strategy_weights(
        sample_returns,
        strategy="hrp",
    )

    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0).all()


def test_black_litterman(sample_returns):

    assets = sample_returns.columns

    market_weights = pd.Series(
        [0.5, 0.3, 0.2],
        index=assets,
    )

    P = pd.DataFrame(
        [[1.0, -1.0, 0.0]],
        index=["A_outperforms_B"],
        columns=assets,
    )

    Q = pd.Series({
        "A_outperforms_B": 0.03,
    })

    confidences = pd.Series({
        "A_outperforms_B": 0.60,
    })

    weights = compute_strategy_weights(
        sample_returns,
        strategy="black_litterman",
        market_weights=market_weights,
        risk_aversion=2.5,
        P=P,
        Q=Q,
        confidences=confidences,
        tau=0.05,
    )

    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= -1e-8).all()
    assert weights.index.equals(assets)


def test_invalid_strategy(sample_returns):

    with pytest.raises(
        ValueError,
        match="unsupported strategy",
    ):
        compute_strategy_weights(
            sample_returns,
            strategy="invalid",
        )


def test_black_litterman_requires_views(sample_returns):

    with pytest.raises(
        ValueError,
        match="requires",
    ):
        compute_strategy_weights(
            sample_returns,
            strategy="black_litterman",
        )


def test_rejects_missing_returns(sample_returns):

    sample_returns.iloc[0, 0] = np.nan

    with pytest.raises(
        ValueError,
        match="missing",
    ):
        compute_strategy_weights(
            sample_returns,
            strategy="min_variance",
        )


def test_equal_weight_preserves_asset_order(sample_returns):

    reordered = sample_returns[["C", "A", "B"]]

    weights = compute_strategy_weights(
        reordered,
        strategy="equal_weight",
    )

    assert weights.index.tolist() == [
        "C", "A", "B"
    ]


def test_hrp_rejects_max_weight(sample_returns):

    with pytest.raises(
        ValueError,
        match="HRP",
    ):
        compute_strategy_weights(
            sample_returns,
            strategy="hrp",
            max_weight=0.5,
        )