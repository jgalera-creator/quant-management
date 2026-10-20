import numpy as np
import pandas as pd
import pytest

from quantmgmt.optimizers.mean_variance import (
    efficient_frontier,
    estimate_inputs,
    max_sharpe,
    min_variance,
    portfolio_return,
    portfolio_variance,
    portfolio_volatility,
    target_volatility,
)


def test_estimate_inputs_expected_returns():
    returns = pd.DataFrame(
        {
            "Asset_A": [0.01, 0.02, 0.03],
            "Asset_B": [-0.01, 0.01, 0.02],
        }
    )

    expected_returns, _ = estimate_inputs(
        returns,
        periods_per_year=12,
    )

    expected = returns.mean() * 12

    pd.testing.assert_series_equal(
        expected_returns,
        expected,
    )


def test_estimate_inputs_covariance():
    returns = pd.DataFrame(
        {
            "Asset_A": [0.01, 0.02, 0.03],
            "Asset_B": [-0.01, 0.01, 0.02],
        }
    )

    _, covariance = estimate_inputs(
        returns,
        periods_per_year=12,
    )

    expected = returns.cov() * 12

    pd.testing.assert_frame_equal(
        covariance,
        expected,
    )


def test_covariance_matrix_is_symmetric():
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.01, 0.03],
            "B": [0.02, -0.01, 0.01, 0.04],
            "C": [-0.01, 0.00, 0.02, 0.01],
        }
    )

    _, covariance = estimate_inputs(returns)

    assert np.allclose(
        covariance.values,
        covariance.values.T,
    )


def test_estimate_inputs_uses_common_sample():
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.03, 0.04],
            "B": [0.01, np.nan, 0.02, 0.03],
        }
    )

    expected_returns, covariance = estimate_inputs(
        returns,
        periods_per_year=12,
    )

    clean = returns.dropna()

    pd.testing.assert_series_equal(
        expected_returns,
        clean.mean() * 12,
    )

    pd.testing.assert_frame_equal(
        covariance,
        clean.cov() * 12,
    )



def test_estimate_inputs_invalid_input():
    with pytest.raises(TypeError):
        estimate_inputs(
            [0.01, 0.02]
        )


def test_estimate_inputs_invalid_periods():
    returns = pd.DataFrame({
        "A": [0.01, 0.02],
    })

    with pytest.raises(ValueError):
        estimate_inputs(
            returns,
            periods_per_year=0,
        )


def test_estimate_inputs_requires_two_observations():
    returns = pd.DataFrame({
        "A": [0.01],
        "B": [0.02],
    })

    with pytest.raises(ValueError):
        estimate_inputs(returns)


def test_portfolio_return():
    expected_returns = pd.Series(
        {
            "A": 0.10,
            "B": 0.20,
        }
    )

    weights = np.array([
        0.25,
        0.75,
    ])

    result = portfolio_return(
        weights,
        expected_returns,
    )

    assert np.isclose(
        result,
        0.175,
    )


def test_portfolio_volatility():
    covariance = pd.DataFrame(
        [
            [0.04, 0.00],
            [0.00, 0.09],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )

    weights = np.array([
        0.5,
        0.5,
    ])

    result = portfolio_volatility(
        weights,
        covariance,
    )

    expected = np.sqrt(
        weights.T
        @ covariance.to_numpy()
        @ weights
    )

    assert np.isclose(
        result,
        expected,
    )


def test_max_sharpe_weights_sum_to_one():
    expected_returns = pd.Series(
        {
            "A": 0.08,
            "B": 0.15,
        }
    )

    covariance = pd.DataFrame(
        [
            [0.04, 0.01],
            [0.01, 0.09],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )

    weights = max_sharpe(
        expected_returns,
        covariance,
    )

    assert np.isclose(
        weights.sum(),
        1.0,
    )


def test_max_sharpe_long_only():
    expected_returns = pd.Series(
        {
            "A": 0.08,
            "B": 0.15,
        }
    )

    covariance = pd.DataFrame(
        [
            [0.04, 0.01],
            [0.01, 0.09],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )

    weights = max_sharpe(
        expected_returns,
        covariance,
        long_only=True,
    )

    assert (weights >= 0).all()
    assert (weights <= 1).all()


def test_max_sharpe_prefers_better_risk_adjusted_asset():
    expected_returns = pd.Series(
        {
            "Good": 0.20,
            "Poor": 0.05,
        }
    )

    covariance = pd.DataFrame(
        [
            [0.04, 0.00],
            [0.00, 0.04],
        ],
        index=["Good", "Poor"],
        columns=["Good", "Poor"],
    )

    weights = max_sharpe(
        expected_returns,
        covariance,
    )

    assert (
        weights["Good"]
        > weights["Poor"]
    )


def test_max_sharpe_respects_max_weight():
    expected_returns = pd.Series(
        {
            "A": 0.20,
            "B": 0.15,
            "C": 0.10,
            "D": 0.08,
            "E": 0.06,
        }
    )

    covariance = pd.DataFrame(
        np.eye(5) * 0.04,
        index=expected_returns.index,
        columns=expected_returns.index,
    )

    weights = max_sharpe(
        expected_returns,
        covariance,
        long_only=True,
        max_weight=0.25,
    )

    assert weights.max() <= 0.25 + 1e-8
    assert np.isclose(weights.sum(), 1.0)


def test_max_weight_rejects_infeasible_constraint():
    covariance = pd.DataFrame(
        np.eye(3),
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )

    with pytest.raises(ValueError):
        min_variance(
            covariance,
            long_only=True,
            max_weight=0.20,
        )



def test_efficient_frontier_returns_points():
    expected_returns = pd.Series(
        {
            "A": 0.08,
            "B": 0.12,
            "C": 0.16,
        }
    )

    covariance = pd.DataFrame(
        [
            [0.04, 0.01, 0.01],
            [0.01, 0.05, 0.01],
            [0.01, 0.01, 0.09],
        ],
        index=expected_returns.index,
        columns=expected_returns.index,
    )

    frontier = efficient_frontier(
        expected_returns,
        covariance,
        n_points=10,
    )

    assert len(frontier) > 0

    assert "return" in frontier.columns
    assert "volatility" in frontier.columns

    assert set(
        expected_returns.index
    ).issubset(frontier.columns)


def test_efficient_frontier_weights_sum_to_one():
    expected_returns = pd.Series(
        {
            "A": 0.08,
            "B": 0.12,
            "C": 0.16,
        }
    )

    covariance = pd.DataFrame(
        [
            [0.04, 0.01, 0.01],
            [0.01, 0.05, 0.01],
            [0.01, 0.01, 0.09],
        ],
        index=expected_returns.index,
        columns=expected_returns.index,
    )

    frontier = efficient_frontier(
        expected_returns,
        covariance,
        n_points=10,
    )

    weight_sums = frontier[
        expected_returns.index
    ].sum(axis=1)

    assert np.allclose(
        weight_sums,
        1.0,
        atol=1e-6,
    )


def test_efficient_frontier_respects_max_weight():
    expected_returns = pd.Series(
        {
            "A": 0.08,
            "B": 0.10,
            "C": 0.12,
            "D": 0.14,
            "E": 0.16,
        }
    )

    covariance = pd.DataFrame(
        np.eye(5) * 0.04,
        index=expected_returns.index,
        columns=expected_returns.index,
    )

    frontier = efficient_frontier(
        expected_returns,
        covariance,
        n_points=10,
        max_weight=0.25,
    )

    weights = frontier[
        expected_returns.index
    ]

    assert (
        weights.max().max()
        <= 0.25 + 1e-6
    )



def test_target_volatility_hits_target():
    expected_returns = pd.Series(
        {
            "A": 0.08,
            "B": 0.12,
            "C": 0.16,
        }
    )

    covariance = pd.DataFrame(
        [
            [0.04, 0.01, 0.01],
            [0.01, 0.05, 0.01],
            [0.01, 0.01, 0.09],
        ],
        index=expected_returns.index,
        columns=expected_returns.index,
    )

    target = 0.18

    weights = target_volatility(
        expected_returns,
        covariance,
        target_vol=target,
    )

    achieved = portfolio_volatility(
        weights.to_numpy(),
        covariance,
    )

    assert np.isclose(
        achieved,
        target,
        atol=1e-6,
    )

    assert np.isclose(
        weights.sum(),
        1.0,
    )


def test_target_volatility_respects_max_weight():
    expected_returns = pd.Series(
        {
            "A": 0.08,
            "B": 0.10,
            "C": 0.12,
            "D": 0.14,
            "E": 0.16,
        }
    )

    covariance = pd.DataFrame(
        np.eye(5) * 0.04,
        index=expected_returns.index,
        columns=expected_returns.index,
    )

    weights = target_volatility(
        expected_returns,
        covariance,
        target_vol=0.10,
        long_only=True,
        max_weight=0.25,
    )

    assert weights.max() <= 0.25 + 1e-6
    assert np.isclose(weights.sum(), 1.0)


def test_target_volatility_rejects_too_low_target():
    expected_returns = pd.Series(
        {
            "A": 0.08,
            "B": 0.12,
        }
    )

    covariance = pd.DataFrame(
        [
            [0.04, 0.01],
            [0.01, 0.09],
        ],
        index=expected_returns.index,
        columns=expected_returns.index,
    )

    with pytest.raises(ValueError):
        target_volatility(
            expected_returns,
            covariance,
            target_vol=0.01,
        )


def test_portfolio_variance():
    covariance = pd.DataFrame(
        [
            [0.04, 0.01],
            [0.01, 0.09],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )

    weights = np.array([
        0.5,
        0.5,
    ])

    result = portfolio_variance(
        weights,
        covariance,
    )

    expected = (
        weights.T
        @ covariance.to_numpy()
        @ weights
    )

    assert np.isclose(
        result,
        expected,
    )


def test_max_sharpe_respects_turnover_limit():
    expected_returns = pd.Series(
        {
            "A": 0.20,
            "B": 0.15,
            "C": 0.10,
        }
    )

    covariance = pd.DataFrame(
        np.eye(3) * 0.04,
        index=expected_returns.index,
        columns=expected_returns.index,
    )

    previous_weights = pd.Series(
        {
            "A": 1 / 3,
            "B": 1 / 3,
            "C": 1 / 3,
        }
    )

    weights = max_sharpe(
        expected_returns,
        covariance,
        previous_weights=previous_weights,
        turnover_limit=0.20,
    )

    turnover = np.abs(
        weights - previous_weights
    ).sum()

    assert turnover <= 0.20 + 1e-6
    assert np.isclose(weights.sum(), 1.0)


def test_min_variance_respects_turnover_limit():
    covariance = pd.DataFrame(
        [
            [0.01, 0.00, 0.00],
            [0.00, 0.04, 0.00],
            [0.00, 0.00, 0.09],
        ],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )

    previous_weights = pd.Series(
        {
            "A": 1 / 3,
            "B": 1 / 3,
            "C": 1 / 3,
        }
    )

    weights = min_variance(
        covariance,
        previous_weights=previous_weights,
        turnover_limit=0.10,
    )

    turnover = np.abs(
        weights - previous_weights
    ).sum()

    assert turnover <= 0.10 + 1e-6
    assert np.isclose(weights.sum(), 1.0)


def test_turnover_requires_previous_weights():
    expected_returns = pd.Series(
        {
            "A": 0.10,
            "B": 0.15,
        }
    )

    covariance = pd.DataFrame(
        [
            [0.04, 0.01],
            [0.01, 0.09],
        ],
        index=expected_returns.index,
        columns=expected_returns.index,
    )

    with pytest.raises(ValueError):
        max_sharpe(
            expected_returns,
            covariance,
            turnover_limit=0.20,
        )


def test_previous_weights_must_sum_to_one():
    expected_returns = pd.Series(
        {
            "A": 0.10,
            "B": 0.15,
        }
    )

    covariance = pd.DataFrame(
        [
            [0.04, 0.01],
            [0.01, 0.09],
        ],
        index=expected_returns.index,
        columns=expected_returns.index,
    )

    previous_weights = pd.Series(
        {
            "A": 0.50,
            "B": 0.40,
        }
    )

    with pytest.raises(ValueError):
        max_sharpe(
            expected_returns,
            covariance,
            previous_weights=previous_weights,
            turnover_limit=0.20,
        )


def test_previous_weights_can_have_different_order():
    expected_returns = pd.Series(
        {
            "A": 0.10,
            "B": 0.15,
            "C": 0.12,
        }
    )

    covariance = pd.DataFrame(
        np.eye(3) * 0.04,
        index=expected_returns.index,
        columns=expected_returns.index,
    )

    previous_weights = pd.Series(
        {
            "C": 0.20,
            "A": 0.50,
            "B": 0.30,
        }
    )

    weights = max_sharpe(
        expected_returns,
        covariance,
        previous_weights=previous_weights,
        turnover_limit=0.20,
    )

    aligned_previous = previous_weights.reindex(
        expected_returns.index
    )

    turnover = np.abs(
        weights - aligned_previous
    ).sum()

    assert turnover <= 0.20 + 1e-6
    assert np.isclose(weights.sum(), 1.0)