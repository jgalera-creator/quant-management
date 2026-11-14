import numpy as np
import pandas as pd
import pytest

from quantmgmt.backtesting.metrics import (
    concentration_metrics,
    annualized_turnover,
    evaluate_backtest,
)


def test_equal_weight_concentration():

    weights = pd.Series({
        "A": 0.25,
        "B": 0.25,
        "C": 0.25,
        "D": 0.25,
    })

    result = concentration_metrics(weights)

    assert result["hhi"].iloc[0] == pytest.approx(
        0.25
    )

    assert result[
        "effective_assets"
    ].iloc[0] == pytest.approx(4.0)


def test_single_asset_concentration():

    weights = pd.Series({
        "A": 1.0,
        "B": 0.0,
        "C": 0.0,
    })

    result = concentration_metrics(weights)

    assert result["hhi"].iloc[0] == pytest.approx(
        1.0
    )

    assert result[
        "effective_assets"
    ].iloc[0] == pytest.approx(1.0)


def test_concentration_multiple_dates():

    dates = pd.bdate_range(
        "2024-01-01",
        periods=2,
    )

    weights = pd.DataFrame(
        {
            "A": [0.5, 1.0],
            "B": [0.5, 0.0],
        },
        index=dates,
    )

    result = concentration_metrics(weights)

    np.testing.assert_allclose(
        result["hhi"],
        [0.5, 1.0],
    )

    np.testing.assert_allclose(
        result["effective_assets"],
        [2.0, 1.0],
    )


def test_concentration_rejects_invalid_weights():

    weights = pd.Series({
        "A": 0.8,
        "B": 0.8,
    })

    with pytest.raises(
        ValueError,
        match="sum to one",
    ):
        concentration_metrics(weights)


def test_annualized_turnover():

    turnover = pd.Series(
        [0.0, 0.10, 0.0, 0.20]
    )

    result = annualized_turnover(
        turnover,
        periods_per_year=4,
    )

    assert result == pytest.approx(0.30)


def test_annualized_turnover_rejects_negative():

    turnover = pd.Series(
        [0.0, -0.10, 0.20]
    )

    with pytest.raises(
        ValueError,
        match="negative",
    ):
        annualized_turnover(turnover)


def test_annualized_turnover_zero():

    turnover = pd.Series(
        np.zeros(252)
    )

    result = annualized_turnover(turnover)

    assert result == pytest.approx(0.0)



from quantmgmt.backtesting.engine import (
    simulate_portfolio,
)                                                                                                                                                                                                                                                                                                                                                                                                      


def test_evaluate_backtest_integration():

    dates = pd.bdate_range(
        "2020-01-01",
        periods=500,
    )

    rng = np.random.default_rng(42)

    returns = pd.DataFrame(
        rng.normal(
            loc=0.0003,
            scale=0.01,
            size=(500, 2),
        ),
        index=dates,
        columns=["A", "B"],
    )

    targets = pd.DataFrame(
        {
            "A": [0.6, 0.5],
            "B": [0.4, 0.5],
        },
        index=[
            dates[0],
            dates[250],
        ],
    )

    simulation = simulate_portfolio(
        returns=returns,
        target_weights=targets,
        initial_capital=100_000,
        transaction_cost_bps=10,
    )

    comparison = {
        "weights": {
            "test_strategy": targets,
        },
        "results": {
            "test_strategy": simulation,
        },
        "nav": pd.DataFrame({
            "test_strategy": simulation["nav"],
        }),
        "portfolio_returns": pd.DataFrame({
            "test_strategy": simulation[
                "portfolio_returns"
            ],
        }),
        "turnover": pd.DataFrame({
            "test_strategy": simulation["turnover"],
        }),
        "costs": pd.DataFrame({
            "test_strategy": simulation["costs"],
        }),
    }

    metrics = evaluate_backtest(comparison)

    assert metrics.shape[0] == 1

    assert "cagr" in metrics.columns
    assert "sharpe" in metrics.columns
    assert "max_drawdown" in metrics.columns
    assert "average_hhi" in metrics.columns

    assert metrics.loc[
        "test_strategy",
        "total_costs",
    ] > 0

    assert metrics.loc[
        "test_strategy",
        "average_effective_assets",
    ] > 1